"""Q3：沿用 Q2 訓練集的 5-fold cross-validation，並以最佳參數重訓後評估 test set。"""

import argparse
import csv
import random
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from q2_holdout import (
    PARAMS, SEED, TEST_SIZE, confusion_metrics, find_libsvm, load_samples,
    parse_accuracy, read_labels, read_predictions, write_libsvm,
)

HERE = Path(__file__).resolve().parent
SPLIT_CSV = HERE / "q2_outputs" / "split.csv"
OUT_DIR = HERE / "q3_outputs"
FOLDS = 5
METRIC_KEYS = ["accuracy", "precision", "recall", "fpr", "tnr", "fnr"]
LOG_LINES = []


def log(text=""):
    print(text)
    LOG_LINES.append(text)


def write_log():
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "run_log.txt").write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")


def display_width(text):
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def show_table(headers, rows):
    grid = [[str(c) for c in row] for row in [headers, *rows]]
    widths = [max(display_width(line[i]) for line in grid) for i in range(len(headers))]
    for idx, line in enumerate(grid):
        log("  ".join(c + " " * (w - display_width(c)) for c, w in zip(line, widths)).rstrip())
        if idx == 0:
            log("  ".join("-" * w for w in widths))


def fmt_pct(value):
    return "N/A" if value is None else f"{value * 100:.2f}%"


def run_cmd(cmd, stdout_file=None):
    cmd = [str(c) for c in cmd]
    cmd_text = subprocess.list2cmdline(cmd)
    log("$ " + cmd_text)
    handle = open(stdout_file, "wb") if stdout_file else None
    try:
        proc = subprocess.run(cmd, stdout=handle if handle else subprocess.PIPE,
                              stderr=subprocess.PIPE)
    finally:
        if handle:
            handle.close()
    out = proc.stdout.decode("utf-8", "replace") if handle is None else ""
    err = proc.stderr.decode("utf-8", "replace")
    if out.strip():
        log(out.strip())
    if err.strip():
        log(err.strip())
    if proc.returncode != 0:
        raise RuntimeError(f"指令失敗（返回碼 {proc.returncode}）: {cmd_text}\n{err.strip()}")
    return out


def load_split():
    with open(SPLIT_CSV, encoding="utf-8", newline="") as f:
        return {int(r["rowid"]): (int(r["label"]), r["set"]) for r in csv.DictReader(f)}


def check_split(samples, split):
    by_rid = {s["rid"]: s for s in samples}
    train_rids = sorted(rid for rid, (_, s) in split.items() if s == "train")
    test_rids = sorted(rid for rid, (_, s) in split.items() if s == "test")
    checks = [
        ("split.csv 筆數與 combats 相同", len(split) == len(samples)),
        ("split.csv rowid 與 combats 完全一致", set(split) == set(by_rid)),
        ("split.csv 標籤與資料庫一致", all(split[r][0] == by_rid[r]["label"] for r in split)),
        ("training 40000 筆", len(train_rids) == 40000),
        ("test 10000 筆", len(test_rids) == TEST_SIZE),
    ]
    for name, ok in checks:
        log(f"[驗證] {name}: {'通過' if ok else '失敗'}")
        if not ok:
            raise RuntimeError(f"Q2 切分核對失敗: {name}")
    return [by_rid[r] for r in train_rids], [by_rid[r] for r in test_rids]


def assign_folds(train):
    total = Counter(s["label"] for s in train)
    expected = {c: total[c] / FOLDS for c in (0, 1)}
    groups = defaultdict(list)
    for s in train:
        groups[s["pair"]].append(s)
    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)

    counts = [Counter() for _ in range(FOLDS)]
    folds = {}
    for key in keys:
        add = Counter(s["label"] for s in groups[key])
        best = min(
            range(FOLDS),
            key=lambda f: sum(
                (counts[f][c] + add[c] - expected[c]) ** 2 - (counts[f][c] - expected[c]) ** 2
                for c in (0, 1)
            ),
        )
        for c in (0, 1):
            counts[best][c] += add[c]
        for s in groups[key]:
            folds[s["rid"]] = best + 1
    return folds


def validate_folds(train, folds):
    rids = {f: {r for r, g in folds.items() if g == f} for f in range(1, FOLDS + 1)}
    pairs = {f: {s["pair"] for s in train if folds[s["rid"]] == f} for f in range(1, FOLDS + 1)}
    all_rids = {s["rid"] for s in train}
    checks = [
        ("五折 rowid 兩兩互斥", all(not (rids[a] & rids[b]) for a, b in combinations(rids, 2))),
        ("五折對手組合兩兩互斥", all(not (pairs[a] & pairs[b]) for a, b in combinations(pairs, 2))),
        ("五折合計涵蓋全部 training", set().union(*rids.values()) == all_rids),
        ("每筆 training 恰好屬於一折", sum(len(v) for v in rids.values()) == len(all_rids)),
        ("每折皆有資料", all(rids[f] for f in rids)),
    ]
    for name, ok in checks:
        log(f"[驗證] {name}: {'通過' if ok else '失敗'}")
        if not ok:
            raise RuntimeError(f"五折切分驗證失敗: {name}")


def write_fold_csv(train, folds):
    with open(OUT_DIR / "fold_assignments.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rowid", "label", "fold"])
        for s in sorted(train, key=lambda r: r["rid"]):
            writer.writerow([s["rid"], s["label"], folds[s["rid"]]])


def mean_or_none(values):
    return None if any(v is None for v in values) else sum(values) / len(values)


def summarize(fold_results):
    summaries = []
    for name, c, g in PARAMS:
        rows = [r for r in fold_results if r["group"] == name]
        totals = {k: sum(r[k] for r in rows) for k in ("TN", "FP", "FN", "TP")}
        if sum(totals.values()) != 40000:
            raise RuntimeError(f"組別 {name} 混淆矩陣總和不是 40000")
        means = {f"mean_{k}": mean_or_none([r[k] for r in rows]) for k in METRIC_KEYS}
        summaries.append({"group": name, "C": c, "gamma": g, **totals, **means})
    return summaries


def select_best(summaries):
    best_acc = max(s["mean_accuracy"] for s in summaries)
    tied = [s for s in summaries if s["mean_accuracy"] == best_acc]
    if len(tied) > 1:
        best_rec = max(s["mean_recall"] for s in tied)
        tied = [s for s in tied if s["mean_recall"] == best_rec]
    if len(tied) > 1:
        best_fpr = min(s["mean_fpr"] for s in tied)
        tied = [s for s in tied if s["mean_fpr"] == best_fpr]
    return tied


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(libsvm_dir):
    scale_exe, train_exe, predict_exe = (libsvm_dir / e for e in
                                         ("svm-scale.exe", "svm-train.exe", "svm-predict.exe"))
    for sub in ("", "models", "predictions", "rounds", "final"):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)
    log(f"LIBSVM 目錄: {libsvm_dir}")

    samples = load_samples()
    split = load_split()
    train, test = check_split(samples, split)

    folds = assign_folds(train)
    validate_folds(train, folds)
    write_fold_csv(train, folds)

    log("\n[每折分布]")
    fold_rows = []
    for f in range(1, FOLDS + 1):
        rows = [s for s in train if folds[s["rid"]] == f]
        c1 = sum(s["label"] for s in rows)
        c0 = len(rows) - c1
        fold_rows.append((f"fold{f}", len(rows), c0, fmt_pct(c0 / len(rows)), c1,
                          fmt_pct(c1 / len(rows))))
    show_table(["折", "總筆數", "標籤0筆數", "標籤0比例", "標籤1筆數", "標籤1比例"], fold_rows)

    log("\n[五折驗證]")
    fold_results = []
    for k in range(1, FOLDS + 1):
        val_rows = [s for s in train if folds[s["rid"]] == k]
        fit_rows = [s for s in train if folds[s["rid"]] != k]
        rdir = OUT_DIR / "rounds" / f"fold{k}"
        rdir.mkdir(parents=True, exist_ok=True)
        raw_fit, raw_val = rdir / "fit.libsvm", rdir / "val.libsvm"
        rng, sc_fit, sc_val = rdir / "scale_range.txt", rdir / "fit.scaled", rdir / "val.scaled"
        write_libsvm(raw_fit, fit_rows)
        write_libsvm(raw_val, val_rows)
        run_cmd([scale_exe, "-l", "-1", "-u", "1", "-s", rng, raw_fit], sc_fit)
        run_cmd([scale_exe, "-r", rng, raw_val], sc_val)

        y_true = read_labels(sc_val)
        if y_true != [s["label"] for s in sorted(val_rows, key=lambda r: r["rid"])]:
            raise RuntimeError(f"fold{k} 驗證標籤與資料不一致")
        for name, c, g in PARAMS:
            model = OUT_DIR / "models" / f"M{name}{k}.model"
            pred = OUT_DIR / "predictions" / f"P{name}{k}.txt"
            run_cmd([train_exe, "-s", "0", "-t", "2", "-c", c, "-g", g, "-q", sc_fit, model])
            out = run_cmd([predict_exe, sc_val, model, pred])
            y_pred = read_predictions(pred)
            if len(y_pred) != len(val_rows):
                raise RuntimeError(f"M{name}{k} 預測筆數 {len(y_pred)} 與驗證集 {len(val_rows)} 不一致")
            m = confusion_metrics(y_true, y_pred)
            libsvm_acc = parse_accuracy(out)
            ok = abs(m["accuracy"] * 100 - libsvm_acc) < 0.005
            log(f"[驗證] M{name}{k} 自算 Accuracy 與 svm-predict 一致: {ok}")
            fold_results.append({"group": name, "C": c, "gamma": g, "fold": k, **m})

    fold_table = [{
        "group": r["group"], "C": r["C"], "gamma": r["gamma"], "fold": r["fold"],
        "TN": r["TN"], "FP": r["FP"], "FN": r["FN"], "TP": r["TP"],
        **{k: r[k] for k in METRIC_KEYS},
    } for r in fold_results]
    write_csv(OUT_DIR / "fold_metrics.csv", fold_table)

    summaries = summarize(fold_results)
    write_csv(OUT_DIR / "cv_summary.csv", summaries)

    log("\n[各折逐折結果]")
    show_table(
        ["組別", "折", "TN", "FP", "FN", "TP", "Accuracy", "Precision", "Recall(TPR)", "FPR", "TNR", "FNR"],
        [(r["group"], r["fold"], r["TN"], r["FP"], r["FN"], r["TP"], fmt_pct(r["accuracy"]),
          fmt_pct(r["precision"]), fmt_pct(r["recall"]), fmt_pct(r["fpr"]), fmt_pct(r["tnr"]),
          fmt_pct(r["fnr"])) for r in fold_results],
    )

    log("\n[五折彙整]  混淆矩陣為五折加總，指標為五折算術平均")
    show_table(
        ["組別", "C", "gamma", "TN", "FP", "FN", "TP", "平均Accuracy", "平均Precision",
         "平均Recall(TPR)", "平均FPR", "平均TNR", "平均FNR"],
        [(s["group"], s["C"], s["gamma"], s["TN"], s["FP"], s["FN"], s["TP"],
          fmt_pct(s["mean_accuracy"]), fmt_pct(s["mean_precision"]), fmt_pct(s["mean_recall"]),
          fmt_pct(s["mean_fpr"]), fmt_pct(s["mean_tnr"]), fmt_pct(s["mean_fnr"]))
         for s in summaries],
    )

    tied = select_best(summaries)
    if len(tied) == 1:
        best = tied[0]
        log(f"\n[最佳參數] 組別 {best['group']} (C={best['C']}, gamma={best['gamma']}), "
            f"平均 Accuracy={fmt_pct(best['mean_accuracy'])}")
    else:
        best = tied[0]
        log("\n[最佳參數] 並列: " + ", ".join(t["group"] for t in tied) +
            f"，依 A–E 順序選用 {best['group']}")

    log("\n[最終重訓] 使用完整 40,000 筆 training 與最佳參數")
    fdir = OUT_DIR / "final"
    raw_train, raw_test = fdir / "train.libsvm", fdir / "test.libsvm"
    rng, sc_train, sc_test = fdir / "scale_range.txt", fdir / "train.scaled", fdir / "test.scaled"
    write_libsvm(raw_train, train)
    write_libsvm(raw_test, test)
    run_cmd([scale_exe, "-l", "-1", "-u", "1", "-s", rng, raw_train], sc_train)
    run_cmd([scale_exe, "-r", rng, raw_test], sc_test)

    model = OUT_DIR / "models" / f"M_final_{best['group']}.model"
    pred = OUT_DIR / "predictions" / f"P_final_{best['group']}.txt"
    run_cmd([train_exe, "-s", "0", "-t", "2", "-c", best["C"], "-g", best["gamma"], "-q",
             sc_train, model])
    out = run_cmd([predict_exe, sc_test, model, pred])
    y_true = read_labels(sc_test)
    y_pred = read_predictions(pred)
    if len(y_pred) != TEST_SIZE or len(y_true) != TEST_SIZE:
        raise RuntimeError(f"test 預測筆數 {len(y_pred)} 不等於 {TEST_SIZE}")
    final = confusion_metrics(y_true, y_pred)
    ok = abs(final["accuracy"] * 100 - parse_accuracy(out)) < 0.005
    log(f"[驗證] 自算 Accuracy 與 svm-predict 一致: {ok}")
    if final["TN"] + final["FP"] + final["FN"] + final["TP"] != TEST_SIZE:
        raise RuntimeError("test 混淆矩陣合計不是 10000")

    log("\n[最終 test 結果]")
    show_table(
        ["組別", "C", "gamma", "TN", "FP", "FN", "TP", "Accuracy", "Precision", "Recall(TPR)",
         "FPR", "TNR", "FNR"],
        [(best["group"], best["C"], best["gamma"], final["TN"], final["FP"], final["FN"],
          final["TP"], fmt_pct(final["accuracy"]), fmt_pct(final["precision"]),
          fmt_pct(final["recall"]), fmt_pct(final["fpr"]), fmt_pct(final["tnr"]),
          fmt_pct(final["fnr"]))],
    )
    write_csv(OUT_DIR / "final_test_metrics.csv", [{
        "group": best["group"], "C": best["C"], "gamma": best["gamma"], **final}])
    return 0


def main():
    parser = argparse.ArgumentParser(description="Q3 5-fold cross-validation")
    parser.add_argument("--libsvm-dir", type=Path, default=None,
                        help="包含 svm-scale.exe、svm-train.exe、svm-predict.exe 的資料夾")
    args = parser.parse_args()

    libsvm_dir = find_libsvm(args.libsvm_dir)
    if libsvm_dir is None:
        log("找不到 LIBSVM 執行檔（需要 svm-scale.exe、svm-train.exe、svm-predict.exe）")
        log(f'請執行: python q3_cross_validation.py --libsvm-dir "<包含 svm-*.exe 的資料夾>"')
        write_log()
        return 1
    try:
        return run(libsvm_dir)
    except RuntimeError as e:
        log(f"執行中止: {e}")
        return 1
    finally:
        write_log()


if __name__ == "__main__":
    sys.exit(main())
