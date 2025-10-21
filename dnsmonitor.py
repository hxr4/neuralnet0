import socket, threading, json, time, requests
from dnslib import DNSRecord, QTYPE, RR, A

CFG = json.load(open("config.json"))
UPSTREAM = (CFG.get("UPSTREAM_DNS", "8.8.8.8"), 53)
SCORE_API = f"http://{CFG.get('HOST', '127.0.0.1')}:{CFG.get('SCORE_PORT', 5000)}/score"
WARNING_IP = CFG.get("WARNING_IP", "127.0.0.1")
CACHE_TTL = CFG.get("DNS_CACHE_TTL_SECONDS", 60)

# --- Simple in-memory cache for DNS responses ---
DNS_CACHE = {}
CACHE_LOCK = threading.Lock()


def get_cached_action(domain):
    """Checks for a valid, non-expired cache entry."""
    with CACHE_LOCK:
        if domain in DNS_CACHE:
            entry_time, action = DNS_CACHE[domain]
            if time.time() - entry_time < CACHE_TTL:
                return action
    return None


def set_cache_action(domain, action):
    """Adds or updates a domain in the cache."""
    with CACHE_LOCK:
        DNS_CACHE[domain] = (time.time(), action)


def handle_query(data, addr, sock):
    try:
        d = DNSRecord.parse(data)
        qname = str(d.q.qname)
        domain = qname.rstrip(".")

        # --- Tier 1: Check Cache First ---
        cached_action = get_cached_action(domain)
        if cached_action:
            action = cached_action
            print(f"[dns_monitor] {domain} -> {action} (cached)")
        else:
            # --- Tier 2: Call Scoring API ---
            payload = {"url": domain, "source": "dns_monitor", "client_ip": addr[0]}
            try:
                r = requests.post(SCORE_API, json=payload, timeout=2)
                result = r.json()
                action = result.get("action", "allow")
                # Cache the new result
                set_cache_action(domain, action)
            except Exception as e:
                print(f"[dns_monitor] score api error: {e}")
                action = "allow"  # Fail open
            print(f"[dns_monitor] {domain} -> {action} (live)")

        # If block or quarantine => return A record pointing to WARNING_IP
        if action in ("block", "quarantine"):
            reply = d.reply()
            reply.add_answer(RR(qname, QTYPE.A, rclass=1, ttl=60, rdata=A(WARNING_IP)))
            sock.sendto(reply.pack(), addr)
            return

        # else forward to upstream and send response back
        upstream = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        upstream.settimeout(2)
        upstream.sendto(data, UPSTREAM)
        reply, _ = upstream.recvfrom(4096)
        upstream.close()
        sock.sendto(reply, addr)

    except Exception as e:
        print(f"Error in handle_query: {e}")


def server(listen_ip="0.0.0.0", port=53):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((listen_ip, port))
    print(f"[dns_monitor] listening on {listen_ip}:{port}, upstream={UPSTREAM}, cache_ttl={CACHE_TTL}s")
    try:
        while True:
            data, addr = sock.recvfrom(4096)
            threading.Thread(target=handle_query, args=(data, addr, sock)).start()
    except KeyboardInterrupt:
        print("\nShutting down DNS monitor...")
    finally:
        sock.close()


if __name__ == "__main__":
    server()
