import requests
import pandas as pd
import joblib
import io
import ssl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# ---------- Data Loaders with Robust Error Handling ----------

def load_openphish():
    """Fetches malicious URLs from OpenPhish."""
    try:
        url = "https://openphish.com/feed.txt"
        # Increased timeout for robustness
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()  # Raise an exception for bad status codes
        urls = resp.text.strip().split("\n")
        print(f"[+] Fetched {len(urls)} URLs from OpenPhish.")
        return pd.DataFrame({"url": urls, "label": 1})
    except requests.exceptions.RequestException as e:
        print(f"[!] Error fetching OpenPhish data: {e}")
        return pd.DataFrame(columns=["url", "label"])


def load_abusech():
    """Fetches malicious URLs from Abuse.ch URLhaus."""
    try:
        url = "https://urlhaus.abuse.ch/downloads/csv/"
        # Use requests to fetch data first for better error handling
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        # Specify encoding and handle potential malformed lines
        csv_string = io.StringIO(response.text)

        # NEW FIX: Ignore the file's header entirely and assign column names
        # manually. This is more robust against source file format changes.
        # The URL is consistently the 3rd column.
        col_names = ["id", "dateadded", "url", "url_status", "threat", "tags", "urlhaus_link", "reporter"]

        # FINAL FIX: Switch to the 'python' engine for parsing. It is slower
        # but more robust against malformed CSVs that can cause C engine errors.
        df = pd.read_csv(
            csv_string,
            comment='#',
            header=None,  # Treat file as having no header
            names=col_names,  # Assign our own column names
            usecols=['url'],  # Now we can reliably select the 'url' column
            on_bad_lines="skip",
            engine='python',  # Use the more flexible python engine to prevent tokenizing errors
        )
        df = df.dropna().drop_duplicates()
        df["label"] = 1
        print(f"[+] Fetched {len(df)} URLs from Abuse.ch.")
        return df
    except requests.exceptions.RequestException as e:
        print(f"[!] Error fetching Abuse.ch data: {e}")
        return pd.DataFrame(columns=["url", "label"])


def load_tranco():
    """Fetches benign domains from the Tranco Top 1M list."""
    try:
        url = "https://tranco-list.eu/top-1m.csv.zip"
        # Use requests to handle the network connection and timeout
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        # Read the zip file content directly from memory into pandas
        df = pd.read_csv(io.BytesIO(response.content), compression="zip", header=None, names=["rank", "domain"])
        benign_urls = ["http://" + d for d in df["domain"].tolist()]
        print(f"[+] Fetched {len(benign_urls)} domains from Tranco.")
        return pd.DataFrame({"url": benign_urls, "label": 0})
    except requests.exceptions.RequestException as e:
        print(f"[!] Error fetching Tranco data: {e}")
        return pd.DataFrame(columns=["url", "label"])
    except Exception as e:
        print(f"[!] An error occurred while processing Tranco data: {e}")
        return pd.DataFrame(columns=["url", "label"])


def build_dataset():
    """Builds a combined and shuffled dataset from all sources."""
    print("\n[+] Building dataset from sources...")
    openphish = load_openphish()
    abusech = load_abusech()
    # Using a smaller, more memory-friendly number of benign sites
    tranco = load_tranco().head(25000)

    # Check if we successfully fetched data before proceeding
    if openphish.empty and abusech.empty:
        raise RuntimeError("Failed to fetch any malicious URL datasets. Cannot proceed.")
    if tranco.empty:
        raise RuntimeError("Failed to fetch Tranco benign dataset. Cannot proceed.")

    # Combine the dataframes
    df = pd.concat([openphish, abusech, tranco], ignore_index=True)
    df = df.dropna()
    df = df.drop_duplicates(subset=['url'])
    # Shuffle the dataset for randomness
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ---------- Main Training Pipeline ----------

if __name__ == "__main__":
    try:
        df = build_dataset()
        print("\n[+] Dataset built successfully.")
        print("    Total unique URLs:", df.shape[0])
        print("    Class distribution:\n", df["label"].value_counts())

        # Feature Engineering: Convert URLs into numerical features
        # Reduced max_features to lower memory usage significantly
        print("\n[+] Vectorizing URLs...")
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 6), max_features=15000)
        X = vectorizer.fit_transform(df["url"].astype(str))
        y = df["label"].values

        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        print(f"    Training set size: {X_train.shape[0]}")
        print(f"    Testing set size: {X_test.shape[0]}")

        # Model Training
        print("\n[+] Training MLP Classifier model...")
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", max_iter=300,
                            random_state=42, early_stopping=True, verbose=True)
        clf.fit(X_train, y_train)
        print("[+] Model training complete.")

        # Evaluation
        print("\n[+] Evaluating model performance...")
        pred = clf.predict(X_test)
        print(f"    Accuracy: {accuracy_score(y_test, pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, pred, target_names=['Benign', 'Phish']))

        # Save the trained model and vectorizer for the Flask app
        print("[+] Saving artifacts...")
        joblib.dump(vectorizer, "vectorizer.pkl")
        joblib.dump(clf, "model.pkl")
        print("[+] Successfully saved vectorizer.pkl and model.pkl")

    except ssl.SSLError as e:
        print(f"\n[!] CRITICAL: An SSL certificate error occurred: {e}")
        print(
            "[!] This often happens on macOS. Please run the 'Install Certificates.command' script in your Python installation folder.")
    except RuntimeError as e:
        print(f"\n[!] CRITICAL: A runtime error occurred: {e}")
    except Exception as e:
        print(f"\n[!] CRITICAL: An unexpected error occurred: {e}")

