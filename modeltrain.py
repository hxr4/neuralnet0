import requests
import pandas as pd
import joblib
import io
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from urllib.parse import urlparse
import tldextract
import re
import zipfile


# -------------------------------------------------------------------
# Expert Feature Engineering Function (Unchanged)
# -------------------------------------------------------------------
def extract_features(url):
    features = []
    if not url.startswith(('http://', 'https://')):
        url = "http://" + url
    hostname, path, domain_parts = '', '', tldextract.extract('')
    try:
        parsed_url = urlparse(url)
        domain_parts = tldextract.extract(url)
        hostname = '.'.join(p for p in [domain_parts.subdomain, domain_parts.domain, domain_parts.suffix] if p)
        path = parsed_url.path
    except Exception:
        hostname, path = str(url), ''
    features.extend([
        len(url), len(hostname), url.count('.'), url.count('-'), url.count('_'),
        url.count('/'), url.count('?'), url.count('='), url.count('@'), url.count('&'),
        len(re.findall(r'\d', url)), len(re.findall(r'[a-zA-Z]', url)),
        sum(url.lower().count(word) for word in
            ['login', 'signin', 'verify', 'account', 'update', 'secure', 'banking', 'password', 'confirm']),
        1 if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', hostname) else 0,
        hostname.count('.'), hostname.count('-'),
        len(domain_parts.subdomain.split('.')) if domain_parts.subdomain else 0,
        len(domain_parts.suffix), len(path), path.count('/'), path.count('.'),
        1 if '//' in path else 0
    ])
    return np.array(features, dtype=np.float32)


# -------------------------------------------------------------------
# NEW & IMPROVED Data Loaders (Using stable sources)
# -------------------------------------------------------------------
def load_openphish():
    try:
        url = "https://openphish.com/feed.txt"
        print("[+] Fetching live OpenPhish feed...")
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return pd.DataFrame({"url": resp.text.strip().split("\n"), "label": 1})
    except Exception as e:
        print(f"[!] Warning: Could not fetch OpenPhish. Reason: {e}")
        return pd.DataFrame()


def load_urlhaus_csv():
    try:
        url = "https://urlhaus.abuse.ch/downloads/csv/"
        print("[+] Fetching URLhaus CSV dump...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        csv_string = io.StringIO(response.text)
        col_names = ["id", "date", "url", "status", "threat", "tags", "reporter"]
        df = pd.read_csv(csv_string, comment='#', header=None, names=col_names, usecols=['url'], on_bad_lines="skip",
                         engine='python')
        df["label"] = 1
        return df.dropna().drop_duplicates()
    except Exception as e:
        print(f"[!] Warning: Could not fetch URLhaus CSV. Reason: {e}")
        return pd.DataFrame()


def load_tranco_top_1m():
    try:
        url = "https://tranco-list.eu/top-1m.csv.zip"
        print("[+] Fetching Tranco Top 1M sites for benign data...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        df = pd.read_csv(io.BytesIO(response.content), compression="zip", header=None, names=["rank", "domain"])
        benign_urls = ["http://" + d for d in df["domain"].tolist()]
        return pd.DataFrame({"url": benign_urls, "label": 0})
    except Exception as e:
        print(f"[!] Warning: Could not fetch Tranco data. Reason: {e}")
        return pd.DataFrame()


def build_dataset_and_blocklist():
    print("\n[+] Building dataset from stable sources...")
    openphish = load_openphish()
    urlhaus = load_urlhaus_csv()
    tranco = load_tranco_top_1m()

    malicious_df = pd.concat([openphish, urlhaus], ignore_index=True).dropna().drop_duplicates(subset=['url'])
    if malicious_df.empty:
        raise RuntimeError("FATAL: Failed to download any malicious URLs. Cannot proceed.")

    num_malicious = len(malicious_df)
    print(f"[+] Found {num_malicious} unique malicious URLs.")
    print(f"[+] Balancing dataset with {num_malicious} benign URLs from Tranco list.")

    if len(tranco) < num_malicious:
        raise RuntimeError("FATAL: Not enough benign URLs from Tranco to balance the dataset.")

    benign_balanced = tranco.head(num_malicious)

    print(f"[+] Saving {num_malicious} malicious URLs to blocklist.txt...")
    with open("blocklist.txt", "w") as f:
        f.writelines(url + '\n' for url in malicious_df["url"])

    df = pd.concat([malicious_df, benign_balanced], ignore_index=True)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# -------------------------------------------------------------------
# Main Training & Quantization Pipeline
# -------------------------------------------------------------------
if __name__ == "__main__":
    try:
        df = build_dataset_and_blocklist()
        print(f"\n[+] Total balanced dataset size: {len(df)}")
        print(df['label'].value_counts())

        print("\n[+] Applying expert feature extraction...")
        features = np.array([extract_features(url) for url in df['url']])
        labels = df['label'].values

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42,
                                                            stratify=labels)

        print("\n[+] Applying feature scaling...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print("\n[+] Building and training the TensorFlow model...")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train_scaled, y_train, epochs=50, batch_size=512, validation_data=(X_test_scaled, y_test),
                  callbacks=[early_stopping], verbose=1)

        print("\n[+] Evaluating final model performance...")
        pred_probs = model.predict(X_test_scaled)
        print("\nClassification Report:")
        print(classification_report(y_test, (pred_probs > 0.5).astype(int), target_names=['Benign', 'Phish']))

        # --- NEW: Quantization to TensorFlow Lite ---
        print("\n[+] Converting to TensorFlow Lite for efficiency...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open("phishing_model.tflite", "wb") as f:
            f.write(tflite_model)
        print("[+] Saved quantized model to phishing_model.tflite")

        joblib.dump(scaler, "scaler.pkl")
        print("[+] Successfully saved scaler.pkl and blocklist.txt")
        print("[+] Advanced training complete.")

    except Exception as e:
        print(f"\n[!] CRITICAL: An unexpected error occurred: {e}")

