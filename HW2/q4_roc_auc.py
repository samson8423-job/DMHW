"""Q4：以 LIBSVM -b 1 取得測試集正類機率，並自行實作 ROC 曲線與 AUC。"""

import argparse
import csv
import subprocess
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from q2_holdout import (  # noqa: E402
    TEST_SIZE, confusion_metrics, find_libsvm, load_samples, parse_accuracy,
    read_labels, write_libsvm,
)
from q3_cross_validation import check_split, load_split  # noqa: E402

HERE = Path(__file__).resolve().parent
Q3_DIR = HERE / "q3_outputs"
Q3_FINAL_DIR = Q3_DIR / "final"
Q3_SUMMARY = Q3_DIR / "cv_summary.csv"
Q3_METRICS = Q3_DIR / "final_test_metrics.csv"
OUT_DIR = HERE / "q4_outputs"
POSITIVE = 1
TRAIN_SIZE = 40000
DPI = 300
LOG_LINES = []


def log(text=""):
    print(text)
    LOG_LINES.append(text)


def write_log():
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "run_log.txt").write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")


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


def read_selected_params():
    with open(Q3_METRICS, encoding="utf-8", newline="") as f:
        final = next(csv.DictReader(f))
    with open(Q3_SUMMARY, encoding="utf-8", newline="") as f:
        summaries = list(csv.DictReader(f))
    best = max(summaries, key=lambda r: float(r["mean_accuracy"]))
    if best["group"] != final["group"]:
        raise RuntimeError("cv_summary.csv 的最佳組別與 final_test_metrics.csv 不一致")
    log(f"[參數] Q3 選出組別 {final['group']}: C={final['C']}, gamma={final['gamma']}")
    return final["group"], final["C"], final["gamma"]


def build_scaled_inputs(libsvm_dir):
    scale_exe = libsvm_dir / "svm-scale.exe"
    samples = load_samples()
    split = load_split()
    train, test = check_split(samples, split)
    d = OUT_DIR / "scaled"
    d.mkdir(parents=True, exist_ok=True)
    raw_train, raw_test = d / "train.libsvm", d / "test.libsvm"
    rng = d / "scale_range.txt"
    write_libsvm(raw_train, train)
    write_libsvm(raw_test, test)
    run_cmd([scale_exe, "-l", "-1", "-u", "1", "-s", rng, raw_train], d / "train.scaled")
    run_cmd([scale_exe, "-r", rng, raw_test], d / "test.scaled")
    return d / "train.scaled", d / "test.scaled"


def ensure_scaled_inputs(libsvm_dir):
    names = ("train.scaled", "test.scaled", "scale_range.txt")
    if all((Q3_FINAL_DIR / n).is_file() for n in names):
        log(f"[輸入] 沿用 Q3 最終重訓縮放檔: {Q3_FINAL_DIR}")
        return Q3_FINAL_DIR / "train.scaled", Q3_FINAL_DIR / "test.scaled"
    log("[輸入] Q3 縮放檔不存在，依 Q2 split.csv 重建")
    return build_scaled_inputs(libsvm_dir)


def parse_prob_file(path):
    """回傳每筆 (預測類別, 正類機率)，正類欄位依標頭 labels 查找。"""
    with open(path, encoding="ascii") as f:
        header = f.readline().split()
        if not header or header[0] != "labels":
            raise RuntimeError("機率輸出第一行不是 labels 標頭")
        labels = [int(x) for x in header[1:]]
        if POSITIVE not in labels:
            raise RuntimeError(f"機率標頭 {labels} 中找不到正類 {POSITIVE}")
        idx = labels.index(POSITIVE)
        rows = []
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            probs = [float(x) for x in parts[1:]]
            if len(probs) != len(labels) or abs(sum(probs) - 1.0) > 1e-6:
                raise RuntimeError("機率欄位數或總和異常")
            rows.append((int(float(parts[0])), probs[idx]))
    return rows


def roc_points(y_true, scores):
    if len(y_true) != len(scores):
        raise RuntimeError("真實標籤與分數筆數不一致")
    pos = sum(1 for y in y_true if y == POSITIVE)
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        raise RuntimeError("正類或負類為 0，無法計算 ROC")
    pairs = sorted(zip(scores, y_true), key=lambda t: t[0], reverse=True)
    points = [(0.0, 0.0, None)]
    tp = fp = 0
    i = 0
    while i < len(pairs):
        threshold = pairs[i][0]
        # 同分樣本整批納入後，才記錄一個 ROC 點
        while i < len(pairs) and pairs[i][0] == threshold:
            if pairs[i][1] == POSITIVE:
                tp += 1
            else:
                fp += 1
            i += 1
        points.append((fp / neg, tp / pos, threshold))
    return points, pos, neg


def check_roc(points):
    checks = [
        ("起點為 (0, 0)", points[0][:2] == (0.0, 0.0)),
        ("終點為 (1, 1)", points[-1][:2] == (1.0, 1.0)),
        ("FPR 單調不減", all(a[0] <= b[0] for a, b in zip(points, points[1:]))),
        ("TPR 單調不減", all(a[1] <= b[1] for a, b in zip(points, points[1:]))),
    ]
    for name, ok in checks:
        log(f"[驗證] ROC {name}: {'通過' if ok else '失敗'}")
        if not ok:
            raise RuntimeError(f"ROC 驗證失敗: {name}")


def integrate_roc_area(points):
    """以相鄰 ROC 點的梯形面積累加求 AUC，保留原始浮點精度。"""
    return sum((x1 - x0) * (y1 + y0) / 2
               for (x0, y0, _), (x1, y1, _) in zip(points, points[1:]))


def auc_by_rank_pairs(y_true, scores):
    """以 Mann-Whitney 配對計數獨立核對 AUC（同分計 0.5）。"""
    neg = sorted(s for s, y in zip(scores, y_true) if y != POSITIVE)
    pos = [s for s, y in zip(scores, y_true) if y == POSITIVE]
    total = 0.0
    for p in pos:
        lo = bisect_left(neg, p)
        hi = bisect_right(neg, p)
        total += lo + 0.5 * (hi - lo)
    return total / (len(pos) * len(neg))


def auc_from_scores(y_true, scores):
    points, _, _ = roc_points(y_true, scores)
    return integrate_roc_area(points)


def self_test():
    cases = [
        ("完美排序 AUC = 1.0", [1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], 1.0),
        ("完全反向 AUC = 0.0", [1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9], 0.0),
        ("正負類分數全部相同 AUC = 0.5", [1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5], 0.5),
    ]
    for name, y, s, expected in cases:
        value = auc_from_scores(y, s)
        ok = abs(value - expected) < 1e-12
        log(f"[自我測試] {name}: 實際 {value} {'通過' if ok else '失敗'}")
        if not ok:
            raise RuntimeError(f"自我測試失敗: {name}")


def plot_roc_png(points, auc_value, path, group, c, gamma):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xs, ys, color="tab:blue", linewidth=2, label=f"LIBSVM -b 1 ROC (AUC = {auc_value:.6f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Diagonal (AUC = 0.5)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title(f"ROC Curve, Test Set (group {group}: C={c}, gamma={gamma})")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def run(libsvm_dir):
    train_exe, predict_exe = libsvm_dir / "svm-train.exe", libsvm_dir / "svm-predict.exe"
    for sub in ("models", "predictions"):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)
    log(f"LIBSVM 目錄: {libsvm_dir}")

    self_test()

    group, c, gamma = read_selected_params()
    train_scaled, test_scaled = ensure_scaled_inputs(libsvm_dir)

    train_labels = read_labels(train_scaled)
    if len(train_labels) != TRAIN_SIZE:
        raise RuntimeError(f"train.scaled 筆數 {len(train_labels)} 不等於 {TRAIN_SIZE}")

    split = load_split()
    test_rids = sorted(rid for rid, (_, s) in split.items() if s == "test")
    y_true = read_labels(test_scaled)
    if len(y_true) != TEST_SIZE or len(test_rids) != TEST_SIZE:
        raise RuntimeError(f"test 筆數不等於 {TEST_SIZE}")
    expected = [split[r][0] for r in test_rids]
    log(f"[驗證] test.scaled 標籤與 split.csv test 標籤逐行一致: {y_true == expected}")
    if y_true != expected:
        raise RuntimeError("test.scaled 標籤與 split.csv 不一致")

    log("\n[訓練] -b 1 機率模型")
    model = OUT_DIR / "models" / f"M_prob_{group}.model"
    run_cmd([train_exe, "-s", "0", "-t", "2", "-c", c, "-g", gamma, "-b", "1", "-q",
             train_scaled, model])

    log("\n[預測] svm-predict -b 1")
    prob_file = OUT_DIR / "predictions" / f"P_prob_{group}.txt"
    out = run_cmd([predict_exe, "-b", "1", test_scaled, model, prob_file])
    libsvm_acc = parse_accuracy(out)
    rows = parse_prob_file(prob_file)
    if len(rows) != TEST_SIZE:
        raise RuntimeError(f"機率輸出筆數 {len(rows)} 不等於 {TEST_SIZE}")
    y_pred = [r[0] for r in rows]
    scores = [r[1] for r in rows]
    cm = confusion_metrics(y_true, y_pred)
    ok = abs(cm["accuracy"] * 100 - libsvm_acc) < 0.005
    log(f"[驗證] 自算 Accuracy 與 svm-predict 一致: {ok}")
    if not ok:
        raise RuntimeError("自算 Accuracy 與 svm-predict 不一致")

    write_csv(OUT_DIR / "test_probabilities.csv", ["rowid", "label", "pred_label", "prob_pos"],
              [(rid, y, p[0], repr(p[1])) for rid, y, p in zip(test_rids, y_true, rows)])

    log("\n[ROC] 自行計算")
    points, pos, neg = roc_points(y_true, scores)
    check_roc(points)
    auc_value = integrate_roc_area(points)
    rank_value = auc_by_rank_pairs(y_true, scores)
    rank_ok = abs(auc_value - rank_value) < 1e-9
    log(f"[驗證] 梯形累加 AUC 與配對計數 AUC 一致: {rank_ok} "
        f"(梯形={auc_value!r}, 配對={rank_value!r})")
    if not rank_ok:
        raise RuntimeError("AUC 兩種計算方式不一致")
    log(f"測試筆數={len(y_true)}, 正類={pos}, 負類={neg}, ROC 點數={len(points)}")
    log(f"AUC = {auc_value:.6f}")

    write_csv(OUT_DIR / "roc_points.csv", ["fpr", "tpr", "threshold"],
              [(repr(x), repr(y), "" if t is None else repr(t)) for x, y, t in points])
    write_csv(OUT_DIR / "auc_summary.csv",
              ["group", "C", "gamma", "n_test", "n_pos", "n_neg", "roc_points", "auc",
               "auc_rank_check"],
              [(group, c, gamma, len(y_true), pos, neg, len(points), repr(auc_value),
                repr(rank_value))])
    plot_roc_png(points, auc_value, OUT_DIR / "roc_curve.png", group, c, gamma)
    log(f"[輸出] ROC 圖: {OUT_DIR / 'roc_curve.png'} (DPI={DPI})")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Q4 ROC Curve and AUC")
    parser.add_argument("--libsvm-dir", type=Path, default=None,
                        help="包含 svm-scale.exe、svm-train.exe、svm-predict.exe 的資料夾")
    args = parser.parse_args()

    libsvm_dir = find_libsvm(args.libsvm_dir)
    if libsvm_dir is None:
        log("找不到 LIBSVM 執行檔（需要 svm-scale.exe、svm-train.exe、svm-predict.exe）")
        log('請執行: python q4_roc_auc.py --libsvm-dir "<包含 svm-*.exe 的資料夾>"')
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
