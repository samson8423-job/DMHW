"""Q2 Hold-out：分層群組 80/20 切分、LIBSVM 訓練與五組參數評估。"""

import argparse
import csv
import random
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "pokemon.db"
OUT_DIR = HERE / "q2_outputs"
SEED = 42
TEST_SIZE = 10000
EXES = ["svm-scale.exe", "svm-train.exe", "svm-predict.exe"]
DEFAULT_LIBSVM_DIRS = [HERE / "libsvm" / "windows", Path(r"C:\DMHW2\libsvm\windows")]
PARAMS = [
    ("A", "0.5", "0.03125"),
    ("B", "2", "0.03125"),
    ("C", "8", "0.03125"),
    ("D", "2", "0.0078125"),
    ("E", "2", "0.125"),
]
STAT_COLS = [("HP", "hp"), ("Attack", "atk"), ("Defense", "def"),
             ("Sp. Atk", "spa"), ("Sp. Def", "spd"), ("Speed", "spe")]
FEATURE_NAMES = ["HP差", "Attack差", "Defense差", "Sp. Atk差", "Sp. Def差", "Speed差",
                 "傳說身分差", "雙屬性身分差"]
LEGENDARY_FLAG = {"TRUE": 1, "FALSE": 0}
LOG_LINES = []


def log(text=""):
    print(text)
    LOG_LINES.append(text)


def write_log():
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "run_log.txt").write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")


def display_width(text):
    import unicodedata
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


def legendary_flag(raw):
    return LEGENDARY_FLAG[str(raw).strip().upper()]


def dual_flag(raw):
    return 0 if raw is None or str(raw).strip() == "" else 1


def load_samples():
    select = ["c.rowid AS rid", "c.First_pokemon AS a_id", "c.Second_pokemon AS b_id",
              "c.Winner AS winner"]
    for alias, side in (("p1", "a"), ("p2", "b")):
        for col, key in STAT_COLS:
            select.append(f'{alias}."{col}" AS {side}_{key}')
        select.append(f'{alias}."Legendary" AS {side}_leg')
        select.append(f'{alias}."Type 2" AS {side}_t2')
    sql = (
        f"SELECT {', '.join(select)} FROM combats c "
        'JOIN pokemon p1 ON p1."#" = c.First_pokemon '
        'JOIN pokemon p2 ON p2."#" = c.Second_pokemon '
        "ORDER BY c.rowid"
    )
    db = sqlite3.connect(DB_PATH.as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        rows = db.execute(sql).fetchall()
    finally:
        db.close()

    samples = []
    for row in rows:
        if row["winner"] not in (row["a_id"], row["b_id"]):
            raise RuntimeError(f"rowid {row['rid']} 的 Winner 不是對戰雙方之一")
        features = [row[f"a_{key}"] - row[f"b_{key}"] for _, key in STAT_COLS]
        features.append(legendary_flag(row["a_leg"]) - legendary_flag(row["b_leg"]))
        features.append(dual_flag(row["a_t2"]) - dual_flag(row["b_t2"]))
        samples.append({
            "rid": row["rid"],
            "label": 1 if row["winner"] == row["a_id"] else 0,
            "pair": tuple(sorted((row["a_id"], row["b_id"]))),
            "features": features,
        })
    return samples


def group_split(samples):
    total = len(samples)
    labels = Counter(s["label"] for s in samples)
    need = {1: round(TEST_SIZE * labels[1] / total)}
    need[0] = TEST_SIZE - need[1]

    groups = defaultdict(list)
    for s in samples:
        groups[s["pair"]].append(s)
    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)

    got = {0: 0, 1: 0}
    test_pairs = set()
    for key in keys:
        add = Counter(s["label"] for s in groups[key])
        if all(got[c] + add[c] <= need[c] for c in (0, 1)):
            test_pairs.add(key)
            for c in (0, 1):
                got[c] += add[c]
    if got != need:
        raise RuntimeError(f"無法達到目標類別筆數: 目標 {need}, 實際 {got}")

    train = [s for s in samples if s["pair"] not in test_pairs]
    test = [s for s in samples if s["pair"] in test_pairs]
    return train, test


def validate_split(samples, train, test):
    train_ids = {s["rid"] for s in train}
    test_ids = {s["rid"] for s in test}
    train_pairs = {s["pair"] for s in train}
    test_pairs = {s["pair"] for s in test}
    checks = [
        ("rowid 無交集", not (train_ids & test_ids)),
        ("對手組合無交集", not (train_pairs & test_pairs)),
        ("合計涵蓋所有對戰", train_ids | test_ids == {s["rid"] for s in samples}),
        ("training 40000 筆", len(train) == len(samples) - TEST_SIZE),
        ("test 10000 筆", len(test) == TEST_SIZE),
    ]
    for name, ok in checks:
        log(f"[驗證] {name}: {'通過' if ok else '失敗'}")
        if not ok:
            raise RuntimeError(f"切分驗證失敗: {name}")


def write_split_csv(samples, train, test):
    train_ids = {s["rid"] for s in train}
    with open(OUT_DIR / "split.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rowid", "label", "set"])
        for s in samples:
            writer.writerow([s["rid"], s["label"], "train" if s["rid"] in train_ids else "test"])


def write_libsvm(path, rows):
    with open(path, "w", encoding="ascii") as f:
        for s in sorted(rows, key=lambda r: r["rid"]):
            parts = [str(s["label"])]
            parts += [f"{i}:{v}" for i, v in enumerate(s["features"], start=1) if v != 0]
            f.write(" ".join(parts) + "\n")


def run_cmd(cmd, stdout_file=None):
    log("$ " + subprocess.list2cmdline([str(c) for c in cmd]))
    out_handle = open(stdout_file, "wb") if stdout_file else None
    try:
        proc = subprocess.run(
            [str(c) for c in cmd],
            stdout=out_handle if out_handle else subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    finally:
        if out_handle:
            out_handle.close()
    text_out = proc.stdout.decode("utf-8", "replace") if out_handle is None else ""
    text_err = proc.stderr.decode("utf-8", "replace")
    if text_out.strip():
        log(text_out.strip())
    if text_err.strip():
        log(text_err.strip())
    if proc.returncode != 0:
        log(f"指令失敗，返回碼 {proc.returncode}")
        write_log()
        sys.exit(proc.returncode)
    return text_out


def read_labels(path):
    with open(path, encoding="ascii") as f:
        return [int(float(line.split()[0])) for line in f if line.strip()]


def read_predictions(path):
    with open(path, encoding="ascii") as f:
        return [int(float(line)) for line in f if line.strip()]


def confusion_metrics(y_true, y_pred):
    tn = fp = fn = tp = 0
    for t, p in zip(y_true, y_pred):
        if t == 1:
            if p == 1:
                tp += 1
            else:
                fn += 1
        elif p == 1:
            fp += 1
        else:
            tn += 1
    n = tn + fp + fn + tp
    return {
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": (tp + tn) / n,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "fpr": fp / (fp + tn) if fp + tn else None,
        "tnr": tn / (tn + fp) if tn + fp else None,
        "fnr": fn / (fn + tp) if fn + tp else None,
    }


def parse_accuracy(text):
    for line in text.splitlines():
        if line.startswith("Accuracy ="):
            return float(line.split("=")[1].split("%")[0])
    raise RuntimeError("svm-predict 輸出中找不到 Accuracy")


def select_best(results):
    best_acc = max(r["accuracy"] for r in results)
    tied = [r for r in results if r["accuracy"] == best_acc]
    best_recall = max(r["recall"] for r in tied)
    tied = [r for r in tied if r["recall"] == best_recall]
    best_fpr = min(r["fpr"] for r in tied)
    return [r for r in tied if r["fpr"] == best_fpr]


def find_libsvm(arg_dir):
    candidates = [arg_dir] if arg_dir else DEFAULT_LIBSVM_DIRS
    for d in candidates:
        if all((d / exe).is_file() for exe in EXES):
            return d
    return None


def main():
    parser = argparse.ArgumentParser(description="Q2 Hold-out 實驗")
    parser.add_argument("--libsvm-dir", type=Path, default=None,
                        help="包含 svm-scale.exe、svm-train.exe、svm-predict.exe 的資料夾")
    args = parser.parse_args()

    libsvm_dir = find_libsvm(args.libsvm_dir)
    if libsvm_dir is None:
        searched = [str(args.libsvm_dir)] if args.libsvm_dir else [str(d) for d in DEFAULT_LIBSVM_DIRS]
        log("找不到 LIBSVM 執行檔（需要 " + ", ".join(EXES) + "）")
        log("已搜尋: " + "; ".join(searched))
        log(f'請執行: python q2_holdout.py --libsvm-dir "<包含 svm-*.exe 的資料夾>"')
        write_log()
        return 1

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "models").mkdir(exist_ok=True)
    (OUT_DIR / "predictions").mkdir(exist_ok=True)
    scale_exe, train_exe, predict_exe = (libsvm_dir / e for e in EXES)
    log(f"LIBSVM 目錄: {libsvm_dir}")

    samples = load_samples()
    train, test = group_split(samples)

    log("\n[切分結果]")
    rows = []
    for name, subset in (("全資料", samples), ("training", train), ("test", test)):
        n = len(subset)
        c1 = sum(s["label"] for s in subset)
        c0 = n - c1
        rows.append((name, n, c0, fmt_pct(c0 / n), c1, fmt_pct(c1 / n)))
    show_table(["集合", "總筆數", "標籤0筆數", "標籤0比例", "標籤1筆數", "標籤1比例"], rows)
    validate_split(samples, train, test)
    write_split_csv(samples, train, test)

    log("\n[LIBSVM 特徵編號]")
    show_table(["編號", "特徵"], [(i, n) for i, n in enumerate(FEATURE_NAMES, start=1)])

    raw_train = OUT_DIR / "train.libsvm"
    raw_test = OUT_DIR / "test.libsvm"
    write_libsvm(raw_train, train)
    write_libsvm(raw_test, test)

    range_file = OUT_DIR / "scale_range.txt"
    scaled_train = OUT_DIR / "train.scaled"
    scaled_test = OUT_DIR / "test.scaled"
    log("\n[縮放]")
    run_cmd([scale_exe, "-l", "-1", "-u", "1", "-s", range_file, raw_train], scaled_train)
    run_cmd([scale_exe, "-r", range_file, raw_test], scaled_test)

    log("\n[五組參數訓練與預測]")
    results = []
    y_true = read_labels(scaled_test)
    if len(y_true) != TEST_SIZE:
        raise RuntimeError(f"test 筆數 {len(y_true)} 不等於 {TEST_SIZE}")
    for name, c, g in PARAMS:
        model = OUT_DIR / "models" / f"model_{name}.model"
        pred = OUT_DIR / "predictions" / f"pred_{name}.txt"
        run_cmd([train_exe, "-s", "0", "-t", "2", "-c", c, "-g", g, "-q", scaled_train, model])
        out = run_cmd([predict_exe, scaled_test, model, pred])
        y_pred = read_predictions(pred)
        if len(y_pred) != TEST_SIZE:
            raise RuntimeError(f"組別 {name} 預測筆數 {len(y_pred)} 不等於 {TEST_SIZE}")
        m = confusion_metrics(y_true, y_pred)
        libsvm_acc = parse_accuracy(out)
        log(f"[驗證] 組別 {name} 自算 Accuracy 與 svm-predict 一致: "
            f"{abs(m['accuracy'] * 100 - libsvm_acc) < 0.005}")
        results.append({"name": name, "C": c, "gamma": g, **m})

    log("\n[混淆矩陣與指標]  矩陣排列為 [[TN, FP], [FN, TP]]")
    for r in results:
        log(f"組別 {r['name']} (C={r['C']}, gamma={r['gamma']}): "
            f"[[{r['TN']}, {r['FP']}], [{r['FN']}, {r['TP']}]]")
    show_table(
        ["組別", "C", "gamma", "TN", "FP", "FN", "TP", "Accuracy", "Precision",
         "Recall(TPR)", "FPR", "TNR", "FNR"],
        [(r["name"], r["C"], r["gamma"], r["TN"], r["FP"], r["FN"], r["TP"],
          fmt_pct(r["accuracy"]), fmt_pct(r["precision"]), fmt_pct(r["recall"]),
          fmt_pct(r["fpr"]), fmt_pct(r["tnr"]), fmt_pct(r["fnr"])) for r in results],
    )

    with open(OUT_DIR / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "C", "gamma", "TN", "FP", "FN", "TP",
                         "accuracy", "precision", "recall", "fpr", "tnr", "fnr"])
        for r in results:
            writer.writerow([r["name"], r["C"], r["gamma"], r["TN"], r["FP"], r["FN"], r["TP"],
                             r["accuracy"], r["precision"], r["recall"], r["fpr"], r["tnr"], r["fnr"]])

    best = select_best(results)
    log("\n[最佳參數]")
    if len(best) == 1:
        b = best[0]
        log(f"最佳組別: {b['name']} (C={b['C']}, gamma={b['gamma']}), Accuracy={fmt_pct(b['accuracy'])}, "
            f"Recall={fmt_pct(b['recall'])}, FPR={fmt_pct(b['fpr'])}")
    else:
        log("並列最佳: " + ", ".join(f"{b['name']} (C={b['C']}, gamma={b['gamma']})" for b in best))

    write_log()
    return 0


if __name__ == "__main__":
    sys.exit(main())
