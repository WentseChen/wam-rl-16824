"""Prints the results table and the paired (McNemar exact) tests against BC."""
import json, math, os
T = ["handover_mic", "stack_blocks_three", "blocks_ranking_size", "handover_block"]
R = {m: json.load(open(f"results/eval_{m}.json")) for m in ("bc", "action", "video", "joint") if os.path.exists(f"results/eval_{m}.json")}


def mcnemar(b, c):
    n = b + c
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


print("task", *R)
for t in T: print(t, *[f"{sum(filter(None, r[t]['per_seed']))}/{r[t]['n']}" for r in R.values()])
print("total", *[f"{sum(sum(filter(None, r[t]['per_seed'])) for t in T)}/{sum(r[t]['n'] for t in T)}" for r in R.values()])
for m, r in R.items():
    for ref in ("bc", "action", "video"):
        if ref == m or m == "bc" or (ref != "bc" and m != "joint"): continue
        pairs = {t: [(x, y) for x, y in zip(R[ref][t]["per_seed"], r[t]["per_seed"]) if y is not None] for t in T}  # None: not evaluated
        pairs["all"] = sum(pairs.values(), [])
        for t, p in pairs.items():
            g, l = sum(not x and y for x, y in p), sum(x and not y for x, y in p)
            print(f"{m} vs {ref}, {t}: +{g}/-{l} p={mcnemar(g, l):.3f} (n={len(p)})")
