#!/usr/bin/env python3
"""E1 + E2 — the calibration every other target in the platform keys off.

  E1  L0 = unloaded p50 at RPS=1, max_tokens=64. Defines SLO = 3 x L0.
  E2  RPS sweep -> saturation curve, the knee C1, and the p99/p50 tail ratio.

Run against a worker directly (bare process, no router) so nothing sits between
the measurement and the model.

  python3 deploy/aws/e1e2-run.py http://<ip>:8001
"""

from __future__ import annotations

import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

PROMPT = "The capital of France is"
MAX_TOKENS = 64


def complete(base: str, timeout: float = 120.0) -> tuple[float, int, str | None]:
    """One completion -> (seconds, completion_tokens, error)."""
    req = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(
            {"prompt": PROMPT, "max_tokens": MAX_TOKENS, "temperature": 0.8, "top_k": 200}
        ).encode(),
        headers={"content-type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.load(r)
        return time.monotonic() - t0, body["usage"]["completion_tokens"], None
    except urllib.error.HTTPError as e:
        return time.monotonic() - t0, 0, f"http {e.code}"
    except Exception as e:
        return time.monotonic() - t0, 0, type(e).__name__


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(int(round(p / 100 * (len(s) - 1))), len(s) - 1)
    return s[i]


def e1(base: str, n: int = 12) -> dict:
    """Unloaded latency. One request at a time — no queueing, no contention."""
    print(f"E1  calibration: {n} sequential requests, {MAX_TOKENS} tokens")
    complete(base)  # warm: first call pays CUDA context + kernel autotune
    lat, toks = [], []
    for _ in range(n):
        s, t, err = complete(base)
        if err:
            print(f"    error: {err}")
            continue
        lat.append(s * 1000)
        toks.append(t)
    L0 = pct(lat, 50)
    out = {
        "n": len(lat),
        "L0_ms": round(L0, 1),
        "p90_ms": round(pct(lat, 90), 1),
        "min_ms": round(min(lat), 1),
        "max_ms": round(max(lat), 1),
        "tokens_per_request": statistics.mode(toks) if toks else 0,
        "tokens_per_second": round(statistics.mean(toks) / (L0 / 1000), 1) if L0 else 0,
        "slo_p99_ms": round(3 * L0, 1),
    }
    print(f"    L0 (p50)     {out['L0_ms']} ms")
    print(f"    decode rate  {out['tokens_per_second']} tok/s")
    print(f"    SLO = 3xL0   p99 <= {out['slo_p99_ms']} ms")
    return out


def offered_load(base: str, rps: float, duration: float) -> dict:
    """Open loop: dispatch on a fixed schedule regardless of how fast the server
    replies. A closed loop would throttle itself to the server's pace and could
    never show saturation."""
    lat: list[float] = []
    errs = 0
    toks = 0
    lock = threading.Lock()
    threads: list[threading.Thread] = []

    def one():
        nonlocal errs, toks
        s, t, err = complete(base)
        with lock:
            if err:
                errs += 1
            else:
                lat.append(s * 1000)
                toks += t

    start = time.monotonic()
    n = int(rps * duration)
    for i in range(n):
        due = start + i / rps
        now = time.monotonic()
        if due > now:
            time.sleep(due - now)
        th = threading.Thread(target=one, daemon=True)
        th.start()
        threads.append(th)
    for th in threads:
        th.join(timeout=180)
    elapsed = time.monotonic() - start

    ok = len(lat)
    return {
        "offered_rps": rps,
        "achieved_rps": round(ok / elapsed, 2),
        "ok": ok,
        "errors": errs,
        "p50_ms": round(pct(lat, 50), 1),
        "p95_ms": round(pct(lat, 95), 1),
        "p99_ms": round(pct(lat, 99), 1),
        "tokens_per_second": round(toks / elapsed, 1),
    }


def main() -> int:
    base = sys.argv[1].rstrip("/")
    info = json.load(urllib.request.urlopen(base + "/stats", timeout=15))
    print(f"worker: model={info.get('model')} device={info.get('device')}")
    if info.get("device") != "cuda":
        print("  REFUSING: worker is not on CUDA — numbers would be meaningless")
        return 1
    print()

    results: dict = {"worker": info}
    results["e1"] = e1(base)
    slo = results["e1"]["slo_p99_ms"]

    print(f"\nE2  saturation sweep (SLO p99 <= {slo} ms)")
    print(f"    {'rps':>5} {'achieved':>9} {'p50':>8} {'p95':>8} {'p99':>9} {'tok/s':>8}  {'errs':>5}")
    points = []
    knee = None
    for rps in (1, 2, 4, 6, 8, 12, 16, 24):
        r = offered_load(base, rps, duration=20)
        points.append(r)
        held = r["errors"] == 0 and r["p99_ms"] <= slo
        print(f"    {rps:>5} {r['achieved_rps']:>9} {r['p50_ms']:>8} {r['p95_ms']:>8} "
              f"{r['p99_ms']:>9} {r['tokens_per_second']:>8}  {r['errors']:>5}"
              f"{'' if held else '   <- over SLO'}")
        if held:
            knee = rps
        else:
            break  # stop at the first violation: capacity is the leading run
    results["e2"] = {"points": points, "knee_rps": knee, "slo_p99_ms": slo}

    peak = max((p["tokens_per_second"] for p in points), default=0)
    # g5.xlarge on-demand, us-east-2
    cost_per_hr = 1.006
    results["cost"] = {
        "instance_usd_per_hour": cost_per_hr,
        "peak_tokens_per_second": peak,
        "usd_per_1m_tokens": round(cost_per_hr / (peak * 3600) * 1e6, 3) if peak else None,
    }
    print(f"\n    C1 (knee)          {knee} rps")
    print(f"    peak throughput    {peak} tok/s")
    print(f"    $/1M tokens        ${results['cost']['usd_per_1m_tokens']}")

    with open(".fleet/e1e2-results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote .fleet/e1e2-results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
