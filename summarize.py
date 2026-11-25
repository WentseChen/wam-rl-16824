"""Prints the results table and the paired (McNemar exact) tests against BC."""
import json, math, os
T = ["handover_mic", "stack_blocks_three", "blocks_ranking_size", "handover_block"]
R = {m: json.load(open(f"results/eval_{m}.json")) for m in ("bc", "action", "video", "joint") if os.path.exists(f"results/eval_{m}.json")}


def mcnemar(b, c):
    n = b + c
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


print("task", *R)
for t in T: print(t, *[f"{sum(r[t]['per_seed'])}/{r[t]['n']}" for r in R.values()])
print("total", *[f"{sum(sum(r[t]['per_seed']) for t in T)}/{sum(r[t]['n'] for t in T)}" for r in R.values()])
for m, r in R.items():
    if m == "bc": continue
    pairs = {t: list(zip(R["bc"][t]["per_seed"], r[t]["per_seed"])) for t in T}
    pairs["all"] = sum(pairs.values(), [])
    for t, p in pairs.items():
        g, l = sum(not x and y for x, y in p), sum(x and not y for x, y in p)
        print(f"{m} vs BC, {t}: +{g}/-{l} p={mcnemar(g, l):.3f}")
