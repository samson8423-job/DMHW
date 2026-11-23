# DMHW2：資料探勘作業 2 — 寶可夢對戰勝負預測

本專案以 `pokemon.db` 的對戰紀錄為資料來源，依序完成探索性分析（Q1）、Hold-out 切分與五組 SVM 參數比較（Q2）、5-fold cross-validation 與最終模型（Q3）、ROC/AUC 分析（Q4），以及預測結果分析（Q5）。模型為 LIBSVM 的 C-SVC（RBF 核，`-s 0 -t 2`），預測「第一隻寶可夢是否獲勝」。

## 環境需求

- Python 3.11（本專案以此版本執行）
- 僅使用標準函式庫（`sqlite3`、`csv`、`subprocess`、`random` 等）
- Q4 另需 `matplotlib`（用於繪製 ROC 曲線，執行時使用 Agg 後端）
- LIBSVM Windows 執行檔：`svm-scale.exe`、`svm-train.exe`、`svm-predict.exe`

LIBSVM 執行檔的搜尋順序：

1. 專案內的 `libsvm\windows\`
2. `C:\DMHW2\libsvm\windows\`

若放在其他位置，Q2、Q3、Q4 可用 `--libsvm-dir` 指定資料夾。Q5 不呼叫 LIBSVM，因此不需要此參數。

## 專案結構

### 資料

| 檔案 | 說明 |
| --- | --- |
| `pokemon.db` | SQLite 資料庫，包含 `combats`（對戰紀錄：`First_pokemon`、`Second_pokemon`、`Winner`）與 `pokemon`（能力值、屬性、傳說身分等） |
| `libsvm\windows\` | LIBSVM Windows 執行檔 |

### 腳本與結果

| 題目 | 腳本 | 主要輸出 | 結果說明 |
| --- | --- | --- | --- |
| Q1 | `q1_explore.py` | 終端機輸出 | [q1_explore_result.md](q1_explore_result.md) |
| Q2 | `q2_holdout.py` | `q2_outputs/`（`split.csv`、`metrics.csv`、模型與預測檔） | [q2_holdout_result.md](q2_holdout_result.md) |
| Q3 | `q3_cross_validation.py` | `q3_outputs/`（`cv_summary.csv`、`final_test_metrics.csv`、`final/`） | [q3_cv_result.md](q3_cv_result.md) |
| Q4 | `q4_roc_auc.py` | `q4_outputs/`（`auc_summary.csv`、`roc_curve.png`、`test_probabilities.csv`） | [q4_roc_result.md](q4_roc_result.md) |
| Q5 | `q5_prediction_analysis.py` | `q5_outputs/`（速度分組、信心分組、逐筆預測表） | [q5_prediction_result.md](q5_prediction_result.md) |

### 其他

| 檔案 | 說明 |
| --- | --- |
| `demo.model`、`demo.pred`、`demo.range`、`demo.scale` | LIBSVM 示範檔，本專案的腳本不會讀取或產生這些檔案 |

## 使用方式

請在專案根目錄依序執行（後續題目依賴前一題的輸出）：

```bash
python q1_explore.py
python q2_holdout.py
python q3_cross_validation.py
python q4_roc_auc.py
python q5_prediction_analysis.py
```

若 LIBSVM 不在預設位置：

```bash
python q2_holdout.py --libsvm-dir "<包含 svm-*.exe 的資料夾>"
python q3_cross_validation.py --libsvm-dir "<包含 svm-*.exe 的資料夾>"
python q4_roc_auc.py --libsvm-dir "<包含 svm-*.exe 的資料夾>"
```

每個腳本執行時會輸出 `run_log.txt`，記錄指令、LIBSVM 輸出與驗證結果。

## 特徵與標籤

- 特徵共 8 個，皆為「第一隻減第二隻」的差值：HP、Attack、Defense、Sp. Atk、Sp. Def、Speed、傳說身分（Legendary）、雙屬性身分（是否有 Type 2）。
- 標籤：第一隻獲勝為 1，否則為 0。
- 切分：分層群組切分，以對手組合為單位，隨機種子 42，training 40,000 筆、test 10,000 筆。同一組對手不會同時出現在 training 與 test。

## 主要結果

| 題目 | 結果 |
| --- | --- |
| Q1 | 非同速的 48,672 場中，速度較快的一方獲勝 45,699 場，勝率 93.89% |
| Q2 | 五組參數中，組別 E（C=2, gamma=0.125）的 Accuracy 最高，為 90.86% |
| Q3 | 5-fold CV 選出組別 E；以全部 training 重訓後，test Accuracy 90.86%，FP 455、FN 459 |
| Q4 | 組別 E 以 `-b 1` 取得機率，test AUC 0.926099，以 0.5 為門檻的 Accuracy 為 90.82% |
| Q5 | test 共 914 筆錯分（Q3 模型）。速度差 31 以上的錯分全部是較慢的一方獲勝；信心最高區間（0.90–1.00）仍有 352 筆錯分 |

Q5 另發現：完整 50,000 場中，1,328 場同速對戰全部由第二隻獲勝。此為資料特性，成因尚未從資料來源確認。

## 注意事項

- 所有 Q3、Q4、Q5 的數字皆由腳本輸出，報告中的數字需與 `run_log.txt` 或對應 CSV 一致。
- Q4 的 AUC 衡量排序能力，與 Q5 的信心校準分析是不同面向，不宜混用。
- Q5 腳本會檢查 Q2、Q3、Q4 輸出的對齊（rowid、標籤、特徵與預測筆數），任一檢查失敗會中止執行。
