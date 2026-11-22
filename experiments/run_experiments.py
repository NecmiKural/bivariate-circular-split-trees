"""
Univariate vs circular bivariate classification trees: full comparison.

Runs on ALL datasets in data/ (not included; see README):
  - tree size: one fit on the full dataset for each learner (no CV needed);
  - accuracy: up to 10 times 10-fold stratified cross-validation, compared with
    the Nadeau & Bengio (2003) corrected resampled paired t-test at alpha=0.05
    (the same test the WEKA Experimenter uses by default).

Outputs (written next to this script):
  - results.json          machine-readable results
  - results_table.txt     formatted table for the report
  - figures/tree_sizes.png, figures/iris_circle.png

Usage:
  python run_experiments.py [budget_seconds_per_dataset]   # default 900

This reuses the Python split finder in python/, so the numbers
match the Java version (identical algorithm). Big datasets can be slow; the CV
budget caps time per dataset and the table records how many folds completed.
"""
import os, sys, json, time, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CFA = os.path.normpath(os.path.join(HERE, ".."))  # repository root
SRC = os.path.join(CFA, "python")
DATA = os.path.join(CFA, "data")
sys.path.insert(0, SRC)


# ---------- helpers ----------
def load_arff(path):
    """Minimal ARFF reader for numeric attributes with a nominal class (last attr)."""
    attrs, rows, in_data = [], [], False
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            low = line.lower()
            if not in_data:
                if low.startswith("@attribute"):
                    rest = line[len("@attribute"):].strip()
                    if rest[0] in "'\"":
                        q = rest[0]
                        name = rest[1:rest.index(q, 1)]
                        typ = rest[rest.index(q, 1) + 1:].strip()
                    else:
                        parts = rest.split(None, 1)
                        name, typ = parts[0], (parts[1].strip() if len(parts) > 1 else "")
                    if typ.startswith("{"):
                        vals = [v.strip().strip("'\"") for v in typ.strip("{} ").split(",")]
                        attrs.append((name, vals))
                    else:
                        attrs.append((name, None))
                elif low.startswith("@data"):
                    in_data = True
            else:
                rows.append([c.strip().strip("'\"") for c in line.split(",")])
    ci = len(attrs) - 1
    cvals = attrs[ci][1] or sorted({r[ci] for r in rows})
    cmap = {v: i for i, v in enumerate(cvals)}
    X, y = [], []
    for r in rows:
        if len(r) <= ci:
            continue
        X.append([float(r[j]) for j in range(ci)])
        y.append(cmap[r[ci]])
    return np.asarray(X, float), np.asarray(y, np.int64), cvals


def corrected_resampled_ttest(diffs, n_train, n_test, alpha=0.05):
    from scipy import stats
    diffs = np.asarray(diffs, float)
    k = len(diffs)
    mean, var = diffs.mean(), diffs.var(ddof=1)
    if var == 0:
        return 0.0, 1.0, mean, False
    cvar = var * (1.0 / k + n_test / n_train)
    t = mean / np.sqrt(cvar)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return t, p, mean, p < alpha


def build(X, y, nc, bivar):
    from ClassificationTree import ClassificationTree
    args = {"num_classes": nc}
    if bivar:
        args["bivariate_split"] = True
    m = ClassificationTree(args)
    m.fit(X, y)
    return m


# ---------- main ----------
def main():
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 900.0
    files = sorted(glob.glob(os.path.join(DATA, "*.arff")),
                   key=lambda p: os.path.getsize(p))  # small first
    results = []
    for path in files:
        name = os.path.basename(path)
        X, y, cnames = load_arff(path)
        nc, (n, d) = len(cnames), X.shape
        print(f"\n=== {name}  (n={n}, d={d}, classes={nc}) ===", flush=True)

        tu, tb = build(X, y, nc, False), build(X, y, nc, True)
        res = {"dataset": name, "n": int(n), "d": int(d), "num_classes": nc,
               "size_univariate": int(tu.tree_size), "size_bivariate": int(tb.tree_size)}
        print(f"  tree size  univariate={tu.tree_size}  bivariate={tb.tree_size}", flush=True)

        # cross-validation
        start = time.time()
        diffs, au_all, ab_all = [], [], []
        n_tr = n_te = 0
        for rep in range(10):
            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=rep)
            for tr, te in skf.split(X, y):
                if time.time() - start > budget:
                    break
                n_tr, n_te = len(tr), len(te)
                mu, mb = build(X[tr], y[tr], nc, False), build(X[tr], y[tr], nc, True)
                au = accuracy_score(y[te], mu.predict(X[te]))
                ab = accuracy_score(y[te], mb.predict(X[te]))
                au_all.append(au); ab_all.append(ab); diffs.append(ab - au)
            else:
                continue
            break
        if diffs:
            t, p, mean, sig = corrected_resampled_ttest(diffs, n_tr, n_te)
            res.update({"cv_folds": len(diffs),
                        "acc_univariate": float(np.mean(au_all)),
                        "acc_bivariate": float(np.mean(ab_all)),
                        "p_value": float(p), "significant": bool(sig),
                        "cv_seconds": round(time.time() - start, 1)})
            print(f"  accuracy    U={np.mean(au_all)*100:.1f}%  B={np.mean(ab_all)*100:.1f}%  "
                  f"p={p:.3f}  folds={len(diffs)}  ({res['cv_seconds']}s)", flush=True)
        results.append(res)
        json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=2)

    write_table(results)
    make_figures(results, X_last=None)
    print("\nDone. See results_table.txt and figures/.")


def write_table(results):
    order = ['iris.arff','hayes-roth.arff','wine.arff','glass.arff','heart-statlog.arff',
             'ecoli.arff','liver-disorders.arff','balance-scale.arff','pima-diabetes.arff',
             'vowel-nocontext.arff','ionosphere.arff','sonar.arff','spectf.arff',
             'vehicle.arff','yeast.arff','phoneme.arff']
    idx = {n: i for i, n in enumerate(order)}
    results = sorted(results, key=lambda r: idx.get(r["dataset"], 999))
    lines = ["%-18s %5s %3s %3s %6s %6s %7s %7s %6s %8s" %
             ("dataset","n","d","K","szU","szB","accU","accB","folds","p")]
    for r in results:
        acc = "acc_univariate" in r
        mark = ""
        if acc and r["significant"]:
            mark = " *" if r["acc_bivariate"] > r["acc_univariate"] else " v"
        lines.append("%-18s %5d %3d %3d %6d %6d %7s %7s %6s %8s%s" % (
            r["dataset"].replace(".arff",""), r["n"], r["d"], r["num_classes"],
            r["size_univariate"], r["size_bivariate"],
            f"{r['acc_univariate']*100:.1f}" if acc else "-",
            f"{r['acc_bivariate']*100:.1f}" if acc else "-",
            str(r.get("cv_folds","-")),
            f"{r['p_value']:.3f}" if acc else "-", mark))
    txt = "\n".join(lines)
    open(os.path.join(HERE, "results_table.txt"), "w").write(txt + "\n")
    print("\n" + txt)


def make_figures(results, X_last):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    figdir = os.path.join(HERE, "figures")
    os.makedirs(figdir, exist_ok=True)

    done = [r for r in results if "acc_univariate" in r]
    if done:
        names = [r["dataset"].replace(".arff","") for r in done]
        u = [r["size_univariate"] for r in done]
        b = [r["size_bivariate"] for r in done]
        x = np.arange(len(names)); w = 0.38
        fig, ax = plt.subplots(figsize=(max(7, len(names)*0.9), 3.6))
        ax.bar(x-w/2, u, w, label="Univariate", color="#4C72B0")
        ax.bar(x+w/2, b, w, label="Bivariate (circular)", color="#C44E52")
        ax.set_ylabel("Tree size (number of nodes)")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(figdir, "tree_sizes.png"), dpi=150); plt.close()

    # iris circle: refit bivariate on iris and read the root split
    iris_path = os.path.join(DATA, "iris.arff")
    if os.path.exists(iris_path):
      try:
        X, y, cn = load_arff(iris_path)
        m = build(X, y, len(cn), True)
        root = m.root
        s = getattr(root, "splitter", None)
        if s is not None and s.x1 is not None:
            a1, a2 = s.attribute1_index, s.attribute2_index
            fig, ax = plt.subplots(figsize=(4.6, 3.6))
            cols = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
            for c in range(len(cn)):
                mm = y == c
                ax.scatter(X[mm, a1], X[mm, a2], c=cols[c % 4], s=16,
                           label=cn[c], edgecolor="k", linewidth=0.3)
            ax.add_patch(Circle((s.x1, s.x2), s.r, fill=False, color="red", lw=1.8))
            ax.plot(s.x1, s.x2, "r+", ms=10, mew=2)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(f"attribute {a1}"); ax.set_ylabel(f"attribute {a2}")
            ax.set_title("Root circular split on iris")
            ax.legend(fontsize=7, loc="upper left")
            plt.savefig(os.path.join(figdir, "iris_circle.png"), dpi=150, bbox_inches="tight")
            plt.close()
      except Exception as e:
        print("  (skipped iris_circle figure:", e, ")")


if __name__ == "__main__":
    main()
