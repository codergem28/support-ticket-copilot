import argparse
import json
import math
from collections import Counter, defaultdict


def percentile(values, p):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


ap = argparse.ArgumentParser()
ap.add_argument("--last", type=int, default=50)
ap.add_argument("--file", default="data/traces.jsonl")
args = ap.parse_args()

traces = [json.loads(line) for line in open(args.file, encoding="utf-8")][-args.last:]
lat = [t["latency"] for t in traces]

print(f"{len(traces)} tickets")
print(f"latency  p50={percentile(lat, .5):.2f}s  p95={percentile(lat, .95):.2f}s  p99={percentile(lat, .99):.2f}s")
print(f"cost     total=${sum(t.get('cost', 0) for t in traces):.6f}  "
      f"per ticket=${sum(t.get('cost', 0) for t in traces) / len(traces):.6f}")

print("\nwhere the time goes (average per ticket):")
per_step = defaultdict(float)
for t in traces:
    for s in t["spans"]:
        per_step[s["name"]] += s["ms"]
total = sum(per_step.values()) or 1
for name, ms in sorted(per_step.items(), key=lambda x: -x[1]):
    print(f"  {name:<10} {ms / len(traces):>8.0f} ms   {ms / total:>5.0%} of span time")

print("\nhow tickets ended (source):")
for source, n in Counter(t.get("source") for t in traces).most_common():
    print(f"  {source}: {n}")

print("\n5 slowest tickets:")
for t in sorted(traces, key=lambda t: -t["latency"])[:5]:
    parts = ", ".join(f"{s['name']} {s['ms']:.0f}ms" for s in t["spans"])
    print(f"  {t['latency']:.2f}s | {t.get('action')} | {parts}")