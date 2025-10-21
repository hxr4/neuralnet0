# dns_monitor_db.py
import socket, threading, json, time, requests
from dnslib import DNSRecord, QTYPE, RR, A
CFG = json.load(open("config.json"))
UPSTREAM = (CFG.get("UPSTREAM_DNS","8.8.8.8"), 53)
SCORE_API = f"http://{CFG.get('HOST','127.0.0.1')}:{CFG.get('SCORE_PORT',5000)}/score"
WARNING_IP = CFG.get("WARNING_IP","127.0.0.1")

def handle_query(data, addr, sock):
    try:
        d = DNSRecord.parse(data)
        qname = str(d.q.qname)
        domain = qname.rstrip(".")
        # Call scoring API (include client_ip if possible)
        payload = {"url": domain, "source": "dns_monitor", "client_ip": addr[0]}
        try:
            r = requests.post(SCORE_API, json=payload, timeout=2)
            result = r.json()
        except Exception as e:
            print("[dns_monitor] score api error:", e)
            result = {"action":"allow"}
        action = result.get("action","allow")
        # If block or quarantine => return A record pointing to WARNING_IP
        if action in ("block","quarantine"):
            reply = DNSRecord()
            q = d.q
            reply.add_question(q)
            # craft an A record answer pointing to WARNING_IP (so browser resolves to our local warning server)
            reply.add_answer(RR(rname=q.qname, rtype=QTYPE.A, rclass=1, ttl=60, rdata=A(WARNING_IP)))
            sock.sendto(reply.pack(), addr)
            print(f"[dns_monitor] {domain} -> {action} (served WARNING_IP)")
            return
        # else forward to upstream and send response back
        upstream = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        upstream.settimeout(2)
        upstream.sendto(data, UPSTREAM)
        reply, _ = upstream.recvfrom(4096)
        upstream.close()
        sock.sendto(reply, addr)
        print(f"[dns_monitor] {domain} -> allow (forwarded)")
    except Exception as e:
        print("Error in handle_query:", e)

def server(listen_ip="0.0.0.0", port=53):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((listen_ip, port))
    print(f"[dns_monitor] listening on {listen_ip}:{port}, upstream={UPSTREAM}, score_api={SCORE_API}")
    try:
        while True:
            data, addr = sock.recvfrom(4096)
            threading.Thread(target=handle_query, args=(data, addr, sock)).start()
    except KeyboardInterrupt:
        print("Shutting down")
    finally:
        sock.close()

if __name__ == "__main__":
    server()