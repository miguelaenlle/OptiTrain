#!/usr/bin/env python3
"""P3.3 — the experiments a single-host cluster cannot run.

Assumes a port-forward to the router on 127.0.0.1:8000.

  a) discovery      router finds pods through the headless Service (heartbeat off)
  b) cross-node     completions are served by pods on DIFFERENT EC2 instances,
                    verified by identity rather than by a 200
  c) black-hole     terminate a whole instance under load. A killed local process
                    replies ECONNREFUSED instantly; a terminating EC2 instance
                    stops answering and its packets are silently dropped, leaving
                    the router with nothing to react to. Bounding that is what
                    ROUTER_CONNECT_TIMEOUT_SECONDS and
                    ROUTER_UPSTREAM_KEEPALIVE=false exist for, and it has never
                    been exercised until now.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

R = "http://127.0.0.1:8000"
REGION = "us-east-2"


def get(path: str, timeout: float = 5.0):
    with urllib.request.urlopen(R + path, timeout=timeout) as r:
        return json.load(r)


def complete(timeout: float = 20.0):
    """One completion. Returns (worker_id, error)."""
    req = urllib.request.Request(
        R + "/v1/completions",
        data=json.dumps({"prompt": "hi", "max_tokens": 8}).encode(),
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r).get("worker_id", "?"), None
    except urllib.error.HTTPError as e:
        return None, f"http {e.code}"
    except Exception as e:
        return None, type(e).__name__


def kubectl(*args) -> str:
    return subprocess.run(
        ["kubectl", *args], capture_output=True, text=True
    ).stdout.strip()


def aws(*args) -> str:
    return subprocess.run(
        ["aws", "--region", REGION, *args], capture_output=True, text=True
    ).stdout.strip()


def upstream_attempts() -> dict[str, int]:
    """Per-worker upstream attempt counts, parsed from the router's /metrics."""
    with urllib.request.urlopen(R + "/metrics", timeout=5) as r:
        body = r.read().decode()
    out: dict[str, int] = {}
    for line in body.splitlines():
        if not line.startswith("fleet_router_upstream_attempts_total{"):
            continue
        labels, _, value = line.partition("} ")
        wid = ""
        for part in labels.split("{", 1)[1].split(","):
            k, _, v = part.partition("=")
            if k == "worker_id":
                wid = v.strip('"')
        if wid:
            out[wid] = out.get(wid, 0) + int(float(value))
    return out


def pod_ip_nodes() -> dict[str, str]:
    """pod IP -> node name, so a router-side worker_id maps to an EC2 instance."""
    out = kubectl(
        "get", "pods", "-l", "app=fleet-worker",
        "-o", "jsonpath={range .items[*]}{.status.podIP} {.spec.nodeName}{'\\n'}{end}",
    )
    return dict(line.split() for line in out.splitlines() if line.strip())


def pod_nodes() -> dict[str, str]:
    out = kubectl(
        "get", "pods", "-l", "app=fleet-worker",
        "-o", "jsonpath={range .items[*]}{.metadata.name} {.spec.nodeName}{'\\n'}{end}",
    )
    return dict(line.split() for line in out.splitlines() if line.strip())


def main() -> int:
    results: dict = {}
    print("=" * 62)

    # --- a) discovery -----------------------------------------------------
    st = get("/fleet/status")
    print(f"(a) DISCOVERY   live_workers={st['live_workers']}")
    for w in st["workers"]:
        print(f"      {w['worker_id']:<18} {w['addr']}")
    results["discovery"] = {"live_workers": st["live_workers"]}
    if st["live_workers"] < 2:
        print("    FAIL: need 2 live workers")
        return 1
    print("    PASS — heartbeat is off, so DNS/Endpoints is the only path")

    # --- b) cross-node ----------------------------------------------------
    nodes = pod_nodes()
    print(f"\n(b) CROSS-NODE  pods span {len(set(nodes.values()))} node(s)")
    for p, n in nodes.items():
        print(f"      {p:<32} {n}")
    # Attribute by the ROUTER's record of which upstream it dialled, not by the
    # upstream's self-report: the stub derives its id from its port, so both
    # pods call themselves "stub-8001" and the response body cannot distinguish
    # them. The router keys on pod IP, which is unique, and it is the more
    # meaningful witness anyway — it is the component doing the choosing.
    before = upstream_attempts()
    for _ in range(20):
        complete()
    after = upstream_attempts()
    served = Counter({ip: after.get(ip, 0) - before.get(ip, 0) for ip in after})
    served = Counter({ip: n for ip, n in served.items() if n > 0})
    print("    router dialled:", dict(served))

    ip_to_node = pod_ip_nodes()
    for ip, n in served.items():
        print(f"      {ip:<14} {n:>3} requests   node={ip_to_node.get(ip, '?')}")
    nodes_used = {ip_to_node.get(ip) for ip in served}
    results["cross_node"] = {
        "by_pod_ip": dict(served),
        "nodes_used": sorted(x for x in nodes_used if x),
    }
    if len(nodes_used - {None}) >= 2:
        print(f"    PASS — traffic served from {len(nodes_used)} DIFFERENT EC2 instances")
    else:
        print("    FAIL — all traffic served from one node")
        return 1

    # --- c) black-hole ----------------------------------------------------
    victim_pod = next(iter(nodes))
    victim_node = nodes[victim_pod]
    victim_ip = kubectl("get", "node", victim_node,
                        "-o", "jsonpath={.status.addresses[?(@.type=='InternalIP')].address}")
    iid = aws("ec2", "describe-instances",
              "--filters", f"Name=private-ip-address,Values={victim_ip}",
              "--query", "Reservations[].Instances[].InstanceId", "--output", "text").split()
    if not iid:
        print("\n(c) BLACK-HOLE  FAIL: could not map node to an instance id")
        return 1
    iid = iid[0]
    print(f"\n(c) BLACK-HOLE  terminating {iid} ({victim_node}) under load")
    print("      a terminating instance stops answering WITHOUT sending RST —")
    print("      the router must time out the connect, not wait on a reply")

    aws("ec2", "terminate-instances", "--instance-ids", iid, "--output", "text")
    t0 = time.monotonic()
    timeline, errors, oks = [], 0, 0
    while time.monotonic() - t0 < 90:
        s = time.monotonic()
        wid, err = complete(timeout=30)
        dt = time.monotonic() - s
        if err:
            errors += 1
        else:
            oks += 1
        timeline.append({"t": round(time.monotonic() - t0, 1), "s": round(dt, 2),
                         "worker": wid, "err": err})
        time.sleep(0.5)

    worst = max(timeline, key=lambda x: x["s"])
    print(f"      requests={len(timeline)} ok={oks} errors={errors}")
    print(f"      slowest request {worst['s']}s at t={worst['t']}s")
    for e in timeline:
        if e["err"] or e["s"] > 2:
            print(f"        t={e['t']:>5}s  {e['s']:>6}s  {e['err'] or e['worker']}")
    st2 = get("/fleet/status")
    print(f"      router: requests={st2['requests']} rerouted={st2['rerouted']} "
          f"failed={st2['failed']} live={st2['live_workers']}")
    results["black_hole"] = {
        "instance": iid, "requests": len(timeline), "ok": oks, "errors": errors,
        "slowest_s": worst["s"], "slowest_at_s": worst["t"],
        "router": {k: st2[k] for k in ("requests", "rerouted", "failed", "live_workers")},
        "timeline": timeline,
    }
    verdict = "PASS" if errors == 0 else f"FAIL — {errors} client-visible errors"
    print(f"    {verdict}")

    with open(".fleet/p33-results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote .fleet/p33-results.json")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
