"""Q5：分析 Q3 最終 E 組與 Q4 機率模型在測試集上的預測結果。"""

import csv
import math
import sqlite3
import sys
from pathlib import Path

import q2_holdout as q2
import q3_cross_validation as q3

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "q5_outputs"
Q3_FINAL = q3.OUT_DIR / "final"
Q3_PRED = q3.OUT_DIR / "predictions" / "P_final_E.txt"
Q3_METRICS = q3.OUT_DIR / "final_test_metrics.csv"
Q4_PROBS = HERE / "q4_outputs" / "test_probabilities.csv"
Q4_PRED = HERE / "q4_outputs" / "predictions" / "P_prob_E.txt"
TEST_SIZE = q2.TEST_SIZE
SPEED_GROUPS = ["0", "1-10", "11-30", "31+"]
CONF_GROUPS = ["[0.50, 0.60)", "[0.60, 0.80)", "[0.80, 0.90)", "[0.90, 1.00]"]
OTHER = "其他"
LOG_LINES = []


def log(text=""):
    print(text)
    LOG_LINES.append(text)


def write_log():
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "run_log.txt").write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")


def check(name, ok):
    log(f"[驗證] {name}: {'通過' if ok else '失敗'}")
    if not ok:
        raise RuntimeError(f"驗證失敗: {name}")


def pct(value):
    return "N/A" if value is None else f"{value * 100:.2f}%"


def show_table(headers, rows):
    grid = [[str(c) for c in row] for row in [headers, *rows]]
    widths = [max(q2.display_width(line[i]) for line in grid) for i in range(len(headers))]
    for idx, line in enumerate(grid):
        log("  ".join(c + " " * (w - q2.display_width(c)) for c, w in zip(line, widths)).rstrip())
        if idx == 0:
            log("  ".join("-" * w for w in widths))


def write_csv(path, headers, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def speed_group(diff):
    if diff == 0:
        return "0"
    if diff <= 10:
        return "1-10"
    if diff <= 30:
        return "11-30"
    return "31+"


def conf_group(conf):
    if 0.5 <= conf < 0.6:
        return CONF_GROUPS[0]
    if 0.6 <= conf < 0.8:
        return CONF_GROUPS[1]
    if 0.8 <= conf < 0.9:
        return CONF_GROUPS[2]
    if 0.9 <= conf <= 1.0:
        return CONF_GROUPS[3]
    return OTHER


def wilson(k, n, z=1.96):
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def load_combats():
    sql = ('SELECT c.rowid AS rid, c.Winner = c.First_pokemon AS first_wins, '
           'p1.Speed AS spe_a, p2.Speed AS spe_b FROM combats c '
           'JOIN pokemon p1 ON p1."#" = c.First_pokemon '
           'JOIN pokemon p2 ON p2."#" = c.Second_pokemon')
    db = sqlite3.connect(q2.DB_PATH.as_uri() + "?mode=ro", uri=True)
    try:
        rows = db.execute(sql).fetchall()
        total = db.execute("SELECT COUNT(*) FROM combats").fetchone()[0]
    finally:
        db.close()
    check("combats 筆數與 Speed JOIN 結果相同", total == len(rows))
    check("Speed 無缺值", all(a is not None and b is not None for _, _, a, b in rows))
    return {rid: {"label": int(fw), "spe_a": a, "spe_b": b} for rid, fw, a, b in rows}


def parse_libsvm_line(line):
    parts = line.split()
    return int(float(parts[0])), {int(k): float(v) for k, v in (p.split(":") for p in parts[1:])}


def parse_prob_file(path):
    with open(path, encoding="ascii") as f:
        header = f.readline().split()
        if not header or header[0] != "labels":
            raise RuntimeError(f"{path} 缺少 labels 標頭")
        labels = [int(x) for x in header[1:]]
        idx = labels.index(1)
        out = []
        for line in f:
            if line.strip():
                parts = line.split()
                out.append((int(float(parts[0])), float(parts[1 + idx])))
    return out


def run():
    OUT_DIR.mkdir(exist_ok=True)

    log("[資料對齊] 讀取 combats、pokemon 與 Q2 切分")
    combats = load_combats()
    log(f"對戰總筆數: {len(combats)}")
    samples = q2.load_samples()
    split = q3.load_split()
    _, test = q3.check_split(samples, split)
    check("test 10000 筆", len(test) == TEST_SIZE)
    test_rids = [s["rid"] for s in test]
    y_true = [s["label"] for s in test]

    log("\n[Q3 對齊] 核對 test.libsvm、test.scaled 與預測檔")
    raw_lines = (Q3_FINAL / "test.libsvm").read_text(encoding="ascii").splitlines()
    raw_ok = len(raw_lines) == TEST_SIZE and all(
        parse_libsvm_line(line) == (s["label"], {i: float(v) for i, v in enumerate(s["features"], 1) if v != 0})
        for line, s in zip(raw_lines, test))
    check("Q3 test.libsvm 逐行與排序後 rowid 的特徵及標籤一致", raw_ok)
    check("Q3 test.scaled 標籤與 split 逐行一致", q2.read_labels(Q3_FINAL / "test.scaled") == y_true)
    pred3 = q2.read_predictions(Q3_PRED)
    check("Q3 預測筆數為 10000", len(pred3) == TEST_SIZE)
    m3 = q2.confusion_metrics(y_true, pred3)
    with open(Q3_METRICS, encoding="utf-8", newline="") as f:
        ref3 = next(csv.DictReader(f))
    check("Q3 重算混淆矩陣與 final_test_metrics.csv 一致",
          int(ref3["FP"]) == m3["FP"] and int(ref3["FN"]) == m3["FN"]
          and abs(float(ref3["accuracy"]) - m3["accuracy"]) < 1e-9)
    q3_match = round(m3["accuracy"] * 100, 2) == 90.86 and m3["FP"] == 455 and m3["FN"] == 459
    log(f"[核對] Q3 與預期 (Accuracy 90.86%, FP 455, FN 459) 相符: {q3_match}")

    log("\n[Q4 對齊] 核對 test_probabilities.csv 與 P_prob_E.txt")
    with open(Q4_PROBS, encoding="utf-8", newline="") as f:
        q4_rows = {int(r["rowid"]): r for r in csv.DictReader(f)}
    check("Q4 rowid 與 test 集完全一致", set(q4_rows) == set(test_rids) and len(q4_rows) == TEST_SIZE)
    check("Q4 標籤與 split 逐行一致",
          all(int(q4_rows[r]["label"]) == y for r, y in zip(test_rids, y_true)))
    prob_file = parse_prob_file(Q4_PRED)
    check("P_prob_E 筆數為 10000", len(prob_file) == TEST_SIZE)
    check("P_prob_E 與 test_probabilities.csv 逐行一致",
          all(int(q4_rows[r]["pred_label"]) == p and abs(float(q4_rows[r]["prob_pos"]) - pp) < 1e-6
              for r, (p, pp) in zip(test_rids, prob_file)))
    pred4 = [int(q4_rows[r]["pred_label"]) for r in test_rids]
    m4 = q2.confusion_metrics(y_true, pred4)
    log(f"[核對] Q4 Accuracy = {pct(m4['accuracy'])} ({sum(p == y for p, y in zip(pred4, y_true))}/{TEST_SIZE}), "
        f"與預期 90.82% 相符: {round(m4['accuracy'] * 100, 2) == 90.82}")

    # 解答一：Q3 最終 E 組
    log("\n========== 解答一：Q3 最終 E 組 ==========")
    rec = []
    for i, s in enumerate(test):
        rid, y = s["rid"], s["label"]
        c = combats[rid]
        a, b = c["spe_a"], c["spe_b"]
        faster = "a" if a > b else ("b" if b > a else "")
        faster_won = "" if not faster else ((y == 1) == (faster == "a"))
        p3 = pred3[i]
        p4 = pred4[i]
        pred_faster_won = "" if not faster else ((p3 == 1) == (faster == "a"))
        prob = float(q4_rows[rid]["prob_pos"])
        conf = prob if p4 == 1 else 1 - prob
        rec.append({
            "rid": rid, "label": y, "pred3": p3, "correct3": p3 == y,
            "fp3": p3 == 1 and y == 0, "fn3": p3 == 0 and y == 1,
            "spe_a": a, "spe_b": b, "diff": abs(a - b), "group": speed_group(abs(a - b)),
            "faster": faster, "faster_won": faster_won, "pred_faster_won": pred_faster_won,
            "pred4": p4, "prob": prob, "conf": conf, "correct4": p4 == y,
            "conf_group": conf_group(conf),
        })

    log("\n[1-1 整體混淆矩陣與指標 (Q3)]")
    show_table(
        ["TN", "FP", "FN", "TP", "Accuracy", "Precision", "Recall(TPR)", "FPR", "TNR", "FNR"],
        [(m3["TN"], m3["FP"], m3["FN"], m3["TP"], pct(m3["accuracy"]), pct(m3["precision"]),
          pct(m3["recall"]), pct(m3["fpr"]), pct(m3["tnr"]), pct(m3["fnr"]))])
    pred1 = m3["TP"] + m3["FP"]
    actual1 = m3["TP"] + m3["FN"]
    log(f"\n預測為 1 的筆數: {pred1}；實際為 1 的筆數: {actual1}；差值 (FP - FN): {m3['FP'] - m3['FN']}")
    log(f"預測為 1 的比例: {pct(pred1 / TEST_SIZE)}；實際為 1 的比例: {pct(actual1 / TEST_SIZE)}")

    log("\n[1-2 絕對速度差分組 (Q3)]")
    speed_rows = []
    for g in SPEED_GROUPS:
        sub = [r for r in rec if r["group"] == g]
        n = len(sub)
        correct = sum(r["correct3"] for r in sub)
        wrong = n - correct
        fp = sum(r["fp3"] for r in sub)
        fn = sum(r["fn3"] for r in sub)
        nontie = [r for r in sub if r["faster"]]
        fw = sum(1 for r in nontie if r["faster_won"])
        slow_wins_wrong = sum(1 for r in nontie if not r["correct3"] and not r["faster_won"])
        pred_fw = sum(1 for r in nontie if r["pred_faster_won"])
        speed_rows.append([g, n, correct, wrong, pct(wrong / n) if n else "N/A", fp, fn,
                           len(nontie), pct(fw / len(nontie)) if nontie else "N/A",
                           pct(pred_fw / len(nontie)) if nontie else "N/A", slow_wins_wrong])
    show_table(["速度差", "場次", "正確", "錯分", "錯誤率", "FP", "FN", "非同速場次",
                "較快方實際勝率", "較快方預測勝率", "錯分中較慢方獲勝"], speed_rows)
    total_n = sum(r[1] for r in speed_rows)
    total_wrong = sum(r[3] for r in speed_rows)
    check("四組合計 10000 場", total_n == TEST_SIZE)
    check("四組合計錯分 914 筆", total_wrong == m3["FP"] + m3["FN"] == 914)
    write_csv(OUT_DIR / "speed_group_errors.csv",
              ["speed_group", "total", "correct", "wrong", "error_rate", "FP", "FN",
               "nontie", "faster_win_rate", "pred_faster_win_rate", "wrong_slower_won"],
              speed_rows)

    all_nontie = [c for c in combats.values() if c["spe_a"] != c["spe_b"]]
    all_fw = sum(1 for c in all_nontie if (c["label"] == 1) == (c["spe_a"] > c["spe_b"]))
    log(f"\n[Q1 對照] 全部 {len(combats)} 場中，非同速 {len(all_nontie)} 場，較快方獲勝 {all_fw} 場，"
        f"勝率 {pct(all_fw / len(all_nontie))}")

    metric_keys = ["TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "fpr", "tnr", "fnr"]
    write_csv(OUT_DIR / "overall_metrics_q3.csv", metric_keys,
              [["N/A" if m3[k] is None else m3[k] for k in metric_keys]])

    # 解答二：Q4 機率模型
    log("\n========== 解答二：Q4 機率模型 (-b 1) 的預測信心 ==========")
    mismatch = sum(1 for r in rec if (r["pred4"] == 1) != (r["prob"] >= 0.5))
    log(f"預測類別與 prob_pos>=0.5 不一致筆數: {mismatch}")
    log(f"Q4 整體 Accuracy: {pct(m4['accuracy'])}；平均預測信心: "
        f"{pct(sum(r['conf'] for r in rec) / TEST_SIZE)}")

    log("\n[2-1 信心分組 (Q4)]")
    conf_rows = []
    bin_total = 0
    for g in CONF_GROUPS + [OTHER]:
        sub = [r for r in rec if r["conf_group"] == g]
        n = len(sub)
        if g != OTHER:
            bin_total += n
        wrong = sum(1 for r in sub if not r["correct4"])
        fp = sum(1 for r in sub if r["pred4"] == 1 and r["label"] == 0)
        fn = sum(1 for r in sub if r["pred4"] == 0 and r["label"] == 1)
        acc = (n - wrong) / n if n else None
        mean_conf = sum(r["conf"] for r in sub) / n if n else None
        lo, hi = wilson(n - wrong, n)
        gap = mean_conf - acc if n else None
        conf_rows.append([g, n, wrong, fp, fn, pct(acc), pct(mean_conf), pct(gap),
                          pct(lo) if n else "N/A", pct(hi) if n else "N/A"])
    show_table(["信心區間", "場次", "錯分", "FP", "FN", "實際準確率", "平均信心", "信心-準確率",
                "準確率 95% Wilson 下界", "上界"], conf_rows)
    check("四個信心區間合計 10000 筆", bin_total == TEST_SIZE)
    write_csv(OUT_DIR / "confidence_bins.csv",
              ["confidence_group", "total", "wrong", "FP", "FN", "accuracy", "mean_confidence",
               "confidence_minus_accuracy", "wilson95_low", "wilson95_high"],
              conf_rows)

    wrong_sorted = sorted((r for r in rec if not r["correct4"]), key=lambda r: -r["conf"])
    log("\n[2-2 信心最高的前 10 筆錯分 (Q4)]")
    show_table(["rowid", "真實標籤", "預測", "prob_pos", "信心", "速度A", "速度B"],
               [[r["rid"], r["label"], r["pred4"], f"{r['prob']:.6f}", f"{r['conf']:.6f}",
                 r["spe_a"], r["spe_b"]] for r in wrong_sorted[:10]])

    write_csv(OUT_DIR / "test_predictions.csv",
              ["rowid", "label", "pred_q3", "correct_q3", "speed_a", "speed_b", "abs_speed_diff",
               "speed_group", "faster_side", "faster_won", "pred_q4", "prob_pos_q4",
               "confidence_q4", "correct_q4", "confidence_group"],
              [[r["rid"], r["label"], r["pred3"], r["correct3"], r["spe_a"], r["spe_b"], r["diff"],
                r["group"], r["faster"], r["faster_won"], r["pred4"], f"{r['prob']:.6f}",
                f"{r['conf']:.6f}", r["correct4"], r["conf_group"]] for r in rec])
    log(f"\n[輸出] {OUT_DIR}")


def main():
    try:
        run()
    except RuntimeError as e:
        log(f"[錯誤] {e}")
        write_log()
        return 1
    write_log()
    return 0


if __name__ == "__main__":
    sys.exit(main())
