"""33-24 Part B on the S22_E4 WAPR pool: k-frame replay of the fixed check tool."""
import json, random, sys, time
from multiprocessing import Pool
sys.path.insert(0, "src")
from compare6d.check import check

E4 = "/mnt/user-data/uploads/KBS_6DPoseDetection/outputs/r33_22/S22_E4.json"
RULES = json.load(open("S24_RULES.json"))
DATASETS = RULES["datasets"]
REV, COMP = RULES["scores"]["expected_reversed"], RULES["scores"]["expected_comparable"]
SCORES = REV + COMP
KS, DRAWS = RULES["k"], RULES["draws_per_cell"]


def load():
    d = json.load(open(E4))
    frames = {}
    for ds in DATASETS:
        per = {}
        for t in d["per_target"][ds]:
            scene, im, obj = t["target"]
            img = (ds, str(scene), str(im))
            for r in t["rows"]:
                if r["status"] != "ok":
                    continue
                per.setdefault(img, []).append((img, str(obj), {s: r[s] for s in SCORES}, bool(r["correct"])))
        frames[ds] = per
    return frames


FRAMES = load()


def rows_for(ds, score, imgs):
    return [(img, obj, sc[score], c) for i in imgs for (img, obj, sc, c) in FRAMES[ds][i]]


def cell(args):
    ds, score, k = args
    rng = random.Random(f"{ds}|{score}|{k}")
    keys = sorted(FRAMES[ds])
    want = "reversed" if score in REV else "comparable"
    wrong = "comparable" if score in REV else "reversed"
    out = {"no_pairs": 0, "correct": 0, "insufficient": 0, "wrong": 0}
    npairs = []
    for i in range(DRAWS):
        imgs = rng.sample(keys, k)
        try:
            res = check(rows_for(ds, score, imgs), draws=2000, seed=i)
        except ValueError:
            out["no_pairs"] += 1
            npairs.append(0)
            continue
        npairs.append(res["pairs"])
        r = res["reading"]
        out["correct" if r == want else "wrong" if r == wrong else "insufficient"] += 1
    npairs.sort()
    q = lambda p: npairs[min(len(npairs) - 1, int(p * len(npairs)))]
    return ds, score, k, {**{key: 100.0 * v / DRAWS for key, v in out.items()},
                          "pairs_median": q(0.5), "pairs_q1": q(0.25), "pairs_q3": q(0.75)}


def full():
    res = {}
    for score in SCORES:
        res[score] = {}
        for ds in DATASETS + ["pooled"]:
            dss = DATASETS if ds == "pooled" else [ds]
            rows = [x for d_ in dss for x in rows_for(d_, score, sorted(FRAMES[d_]))]
            r = check(rows, draws=2000, seed=0)
            r.pop("advice")
            res[score][ds] = r
    return res


if __name__ == "__main__":
    t0 = time.time()
    if sys.argv[1:] == ["full"]:
        json.dump(full(), open("S24_A_wapr.json", "w"), indent=1)
        print("full done", time.time() - t0)
        sys.exit()
    jobs = [(ds, s, k) for s in SCORES for ds in DATASETS for k in KS]
    if sys.argv[1:] == ["time"]:
        t = time.time(); print(cell(("lmo", "fp_single", 20))[3], time.time() - t); sys.exit()
    results = {}
    with Pool(2) as pool:
        for ds, s, k, v in pool.imap_unordered(cell, jobs):
            results.setdefault(s, {}).setdefault(ds, {})[str(k)] = v
            print(ds, s, k, v, flush=True)
    json.dump({"rules_sha256": __import__("hashlib").sha256(open("S24_RULES.json", "rb").read()).hexdigest(),
               "results": results, "wall_seconds": time.time() - t0}, open("S24_B_wapr.json", "w"), indent=1)
