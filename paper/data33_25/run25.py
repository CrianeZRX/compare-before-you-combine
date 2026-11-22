"""33-25: gated check (v1.1) against a negative control, replay and sequential labelling.

CPU only, on rows already delivered in 33-24. Rules: S25_RULES.json (written before results).
Usage: python3 run25.py repro | null | replay | seq | full | all
"""
import json, os, random, sys, time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
# the released check (v1.1, six-image rule): src/ of the repository, or the working copy used for this run
SRC = next(p for p in (HERE.parents[1] / "src", Path("/tmp/k25/src")) if (p / "compare6d" / "check.py").exists())
sys.path.insert(0, str(SRC))
from compare6d.check import check           # noqa: E402

ROWS = HERE.parent / "data33_24" / "S24_B_rows"
# S22_E4.json (33-22 E4, WAPR poses) is not in the repository; set S22_E4 to its path
E4 = Path(os.environ.get("S22_E4", "/mnt/user-data/uploads/KBS_6DPoseDetection/outputs/r33_22/S22_E4.json"))
RULES = json.load(open(HERE / "S25_RULES.json"))
CLASSIC = ["lmo", "tless", "tudl", "icbin", "ycbv"]
KS, DRAWS = [3, 5, 10, 20], 1000
POOLS = RULES["experiments"]["R_replay"]["pools"]
REV = {"fp_raw", "fp_single", "wbps_neg"}


def load_b():
    frames = {}
    for ds in CLASSIC + ["xyzibd"]:
        d = json.load(open(ROWS / f"{ds}.json"))
        per = {(ds, str(s), str(i)): [] for s, i in d["frames"]}
        for r in d["rows"]:
            per[(ds, str(r["scene_id"]), str(r["im_id"]))].append(
                (str(r["obj_id"]), r["scores"], bool(r["correct"])))
        frames[ds] = per
    return frames


def load_e4():
    d = json.load(open(E4))
    scores = POOLS["wapr_33_22_E4"]
    frames = {}
    for ds in CLASSIC:
        per = {}
        for t in d["per_target"][ds]:
            scene, im, obj = t["target"]
            img = (ds, str(scene), str(im))
            for r in t["rows"]:
                if r["status"] != "ok":
                    continue
                per.setdefault(img, []).append((str(obj), {s: r[s] for s in scores}, bool(r["correct"])))
        frames[ds] = per
    return frames


B, W = load_b(), load_e4()


def pool_of(pool):
    return W if pool == "wapr_33_22_E4" else B


def rows_for(pool, ds, score, imgs, flips=None):
    F = pool_of(pool)[ds]
    out = []
    for img in imgs:
        sgn = -1.0 if flips and flips.get(img) else 1.0
        for obj, sc, c in F[img]:
            v = sc.get(score)
            if v is None:
                continue
            out.append((img, obj, sgn * v, c))
    return out


def v10(res):
    if res["ties"] == res["pairs"]:
        return "no_information"
    lo, hi = res["interval_95"]
    return "reversed" if hi < 0.5 else "comparable" if lo > 0.5 else "not_comparable"


def run(rows, seed):
    try:
        res = check(rows, draws=2000, seed=seed)
    except ValueError:
        return None
    return res


def outcome(reading, want):
    wrong = "comparable" if want == "reversed" else "reversed"
    return "correct" if reading == want else "wrong" if reading == wrong else "insufficient"


def q(vals, p):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(p * len(vals)))]


# ---------- cells ----------
def cell_null(args):
    ds, score, k = args
    keys = sorted(B[ds])
    rng = random.Random(f"null|{ds}|{score}|{k}")
    tally = {t: {} for t in ("v1.0", "v1.1")}
    inf = []
    for i in range(DRAWS):
        imgs = rng.sample(keys, k)
        frng = random.Random(f"flip|{ds}|{score}|{k}|{i}")
        flips = {img: frng.random() < 0.5 for img in sorted(imgs)}
        res = run(rows_for("classic_33_19_E1", ds, score, imgs, flips), i)
        if res is None:
            r10 = r11 = "no_pairs"
            inf.append(0)
        else:
            r10, r11 = v10(res), res["reading"]
            inf.append(res["images_with_pairs"])
        for t, r in (("v1.0", r10), ("v1.1", r11)):
            tally[t][r] = tally[t].get(r, 0) + 1
    out = {}
    for t in tally:
        c = tally[t]
        out[t] = {"counts": c, "definite_pct": 100.0 * (c.get("reversed", 0) + c.get("comparable", 0)) / DRAWS,
                  "reversed_pct": 100.0 * c.get("reversed", 0) / DRAWS,
                  "comparable_pct": 100.0 * c.get("comparable", 0) / DRAWS}
    out["informative_images"] = {"median": q(inf, .5), "q1": q(inf, .25), "q3": q(inf, .75),
                                 "share_lt6_pct": 100.0 * sum(x < 6 for x in inf) / DRAWS,
                                 "share_le1_pct": 100.0 * sum(x <= 1 for x in inf) / DRAWS}
    return ("null", ds, score, k), out


def cell_replay(args):
    pool, ds, score, k = args
    F = pool_of(pool)[ds]
    keys = sorted(F)
    rng = random.Random(f"{ds}|{score}|{k}")
    want = "reversed" if score in REV else "comparable"
    tally = {"v1.0": {}, "v1.1": {}}
    inf = []
    for i in range(DRAWS):
        imgs = rng.sample(keys, k)
        res = run(rows_for(pool, ds, score, imgs), i)
        if res is None:
            o10 = o11 = "no_pairs"
            inf.append(0)
        else:
            o10, o11 = outcome(v10(res), want), outcome(res["reading"], want)
            inf.append(res["images_with_pairs"])
        for t, o in (("v1.0", o10), ("v1.1", o11)):
            tally[t][o] = tally[t].get(o, 0) + 1
    out = {t: {o: 100.0 * tally[t].get(o, 0) / DRAWS for o in ("no_pairs", "correct", "insufficient", "wrong")}
           for t in tally}
    out["informative_images"] = {"median": q(inf, .5), "q1": q(inf, .25), "q3": q(inf, .75),
                                 "share_lt6_pct": 100.0 * sum(x < 6 for x in inf) / DRAWS}
    return ("replay", pool, ds, score, k), out


def cell_seq(args):
    pool, ds, score, m = args
    F = pool_of(pool)[ds]
    want = "reversed" if score in REV else "comparable"
    tally, labelled, short = {}, [], 0
    for i in range(DRAWS):
        keys = sorted(F)
        random.Random(f"seq|{pool}|{ds}|{score}|{i}").shuffle(keys)
        taken, informative = [], 0
        for img in keys:
            taken.append(img)
            r = rows_for(pool, ds, score, [img])
            per = {}
            for _, obj, _, c in r:
                per.setdefault(obj, set()).add(c)
            if any(len(v) == 2 for v in per.values()):
                informative += 1
            if informative >= m:
                break
        if informative < m:
            short += 1
        labelled.append(len(taken))
        res = run(rows_for(pool, ds, score, taken), i)
        o = "no_pairs" if res is None else outcome(res["reading"], want)
        tally[o] = tally.get(o, 0) + 1
    out = {o: 100.0 * tally.get(o, 0) / DRAWS for o in ("no_pairs", "correct", "insufficient", "wrong")}
    out["frames_labelled"] = {"median": q(labelled, .5), "p90": q(labelled, .9), "max": max(labelled)}
    out["ran_out_of_frames"] = short
    return ("seq", pool, ds, score, m), out


def cell_full(args):
    pool, ds, score = args
    dss = CLASSIC if ds == "pooled" else [ds]
    rows = [x for d_ in dss for x in rows_for(pool, d_, score, sorted(pool_of(pool)[d_]))]
    res = run(rows, 0)
    if res is None:
        return ("full", pool, ds, score), None
    return ("full", pool, ds, score), {"v1.0": v10(res), "v1.1": res["reading"], "accuracy": res["accuracy"],
                                       "interval_95": res["interval_95"], "sign_flip_p": res["sign_flip_p"],
                                       "images_with_pairs": res["images_with_pairs"]}


def cell_seqnull(args):
    ds, score = args
    F = B[ds]
    tally = {}
    for i in range(DRAWS):
        keys = sorted(F)
        random.Random(f"seqnull|{ds}|{score}|{i}").shuffle(keys)
        frng = random.Random(f"seqflip|{ds}|{score}|{i}")
        taken, flips, informative = [], {}, 0
        for img in keys:
            taken.append(img)
            flips[img] = frng.random() < 0.5
            per = {}
            for _, obj, _, c in rows_for("classic_33_19_E1", ds, score, [img]):
                per.setdefault(obj, set()).add(c)
            if any(len(v) == 2 for v in per.values()):
                informative += 1
            if informative >= 10:
                break
        res = run(rows_for("classic_33_19_E1", ds, score, taken, flips), i)
        r = "no_pairs" if res is None else res["reading"]
        tally[r] = tally.get(r, 0) + 1
    return ("seqnull", ds, score), {"counts": tally,
                                    "definite_pct": 100.0 * (tally.get("reversed", 0) + tally.get("comparable", 0)) / DRAWS}


def datasets_of(pool):
    return ["xyzibd"] if pool == "xyzibd_33_19_E1e_B1" else CLASSIC


def jobs(kind):
    if kind == "null":
        return [(ds, s, k) for ds in CLASSIC + ["xyzibd"] for s in ("fp_raw", "depth") for k in KS], cell_null
    if kind == "replay":
        return [(p, ds, s, k) for p, ss in POOLS.items() for ds in datasets_of(p) for s in ss for k in KS], cell_replay
    if kind == "seq":
        return [(p, ds, s, m) for p, ss in POOLS.items() for ds in datasets_of(p) for s in ss for m in (6, 10)], cell_seq
    if kind == "seqnull":
        return [(ds, s) for ds in CLASSIC + ["xyzibd"] for s in ("fp_raw", "depth")], cell_seqnull
    if kind == "full":
        return [(p, ds, s) for p, ss in POOLS.items() for ds in datasets_of(p) + (["pooled"] if p != "xyzibd_33_19_E1e_B1" else [])
                for s in ss], cell_full
    raise SystemExit(kind)


def main(kind):
    t0 = time.time()
    if kind == "repro":
        for a in [("classic_33_19_E1", "lmo", "fp_raw", 10), ("xyzibd_33_19_E1e_B1", "xyzibd", "fp_raw", 10)]:
            print(a, cell_replay(a)[1])
        return
    js, fn = jobs(kind)
    with Pool(2) as pool:
        res = pool.map(fn, js, chunksize=1)
    out = {"rules_sha256": __import__("hashlib").sha256(open(HERE / "S25_RULES.json", "rb").read()).hexdigest(),
           "amendment_sha256": __import__("hashlib").sha256(open(HERE / "S25_RULES_AMEND.json", "rb").read()).hexdigest(),
           "tool": "check v1.1 (/tmp/k25), v1.0 readings derived from the identical bootstrap interval",
           "cells": [{"key": list(k), "result": v} for k, v in res]}
    json.dump(out, open(HERE / f"S25_{kind}.json", "w"), indent=1)
    print(kind, len(res), "cells", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    for kind in (["null", "replay", "seq", "full"] if sys.argv[1] == "all" else sys.argv[1:]):
        main(kind)
