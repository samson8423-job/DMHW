"""第一題：寶可夢資料探索性分析與候選特徵展示（唯讀，僅使用標準函式庫）。"""

import sqlite3
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "pokemon.db"
STATS = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
FIELD = {
    "Type 1": "type1",
    "Type 2": "type2",
    "Generation": "gen",
    "Legendary": "legendary",
    "HP": "hp",
    "Attack": "attack",
    "Defense": "defense",
    "Sp. Atk": "sp_atk",
    "Sp. Def": "sp_def",
    "Speed": "speed",
}
LEGENDARY_LABEL = {True: "是 (True)", False: "否 (False)", None: "無法判定"}


def connect_readonly():
    db = sqlite3.connect(DB_PATH.as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def display_width(text):
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def print_table(headers, rows):
    grid = [[str(cell) for cell in row] for row in [headers, *rows]]
    widths = [max(display_width(line[i]) for line in grid) for i in range(len(headers))]
    for idx, line in enumerate(grid):
        padded = "  ".join(cell + " " * (w - display_width(cell)) for cell, w in zip(line, widths))
        print(padded.rstrip())
        if idx == 0:
            print("  ".join("-" * w for w in widths))


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def fmt_pct(part, whole):
    return "N/A" if whole == 0 else f"{part / whole * 100:.2f}%"


def fmt_num(value):
    return "N/A" if value is None else f"{value:.2f}"


def is_blank(value):
    return value is None or str(value).strip() == ""


def to_bool(raw):
    return {"TRUE": True, "FALSE": False}.get(str(raw).strip().upper())


def count_blank(db, table, column):
    sql = f'SELECT COUNT(*) FROM "{table}" WHERE "{column}" IS NULL OR TRIM("{column}") = \'\''
    return db.execute(sql).fetchone()[0]


def load_joined(db):
    select = [
        "c.rowid AS combat_id",
        "c.First_pokemon AS first_id",
        "c.Second_pokemon AS second_id",
        "c.Winner AS winner",
    ]
    for alias, side in (("p1", "f"), ("p2", "s")):
        for name, field in FIELD.items():
            select.append(f'{alias}."{name}" AS {side}_{field}')
    sql = (
        f"SELECT {', '.join(select)} FROM combats c "
        'JOIN pokemon p1 ON p1."#" = c.First_pokemon '
        'JOIN pokemon p2 ON p2."#" = c.Second_pokemon '
        "ORDER BY c.rowid"
    )
    records = [dict(row) for row in db.execute(sql).fetchall()]
    for rec in records:
        rec["first_won"] = rec["winner"] == rec["first_id"]
    return records


def check_data(db, total_combats, joined, valid):
    section("1. 資料檢查")
    for table in ("pokemon", "combats"):
        columns = db.execute(f'PRAGMA table_info("{table}")').fetchall()
        count = db.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        print(f"\n[{table}] 筆數: {count}")
        print_table(["cid", "欄名", "型別"], [(c["cid"], c["name"], c["type"]) for c in columns])

    dup = db.execute(
        'SELECT COUNT(*) FROM (SELECT "#" FROM pokemon GROUP BY "#" HAVING COUNT(*) > 1)'
    ).fetchone()[0]
    print(f"\n[pokemon 編號] 重複編號數: {dup}")

    first_missing, second_missing = db.execute(
        'SELECT SUM(p1."#" IS NULL), SUM(p2."#" IS NULL) FROM combats c '
        'LEFT JOIN pokemon p1 ON p1."#" = c.First_pokemon '
        'LEFT JOIN pokemon p2 ON p2."#" = c.Second_pokemon'
    ).fetchone()
    print(
        f"[combats 對應] 第一方找不到: {first_missing or 0} 筆；"
        f"第二方找不到: {second_missing or 0} 筆；"
        f"任一方找不到: {total_combats - len(joined)} 筆"
    )
    print(f"[combats Winner] 非對戰雙方之一（含 NULL）: {len(joined) - len(valid)} 筆")

    missing = []
    for table in ("pokemon", "combats"):
        for col in db.execute(f'PRAGMA table_info("{table}")').fetchall():
            name = col["name"]
            if table == "pokemon" and name == "Type 2":
                continue
            missing.append((f"{table}.{name}", count_blank(db, table, name)))
    print("\n[缺值統計（不含 Type 2）]")
    print_table(["欄位", "缺值數"], missing)
    single = count_blank(db, "pokemon", "Type 2")
    print(f"\n[Type 2] NULL 或空白（單屬性，不視為缺值）: {single} 筆")

    total_pokemon = db.execute("SELECT COUNT(*) FROM pokemon").fetchone()[0]
    print(
        f"\n[實際使用筆數] pokemon: {total_pokemon} 筆（能力統計依各欄有效筆數）；"
        f"combats: {len(valid)} 筆（雙方可對應且 Winner 合法，能力比較依各項缺值排除）"
    )


def describe_pokemon(db):
    section("2. 寶可夢能力與類別分布")
    pokemon = db.execute("SELECT * FROM pokemon").fetchall()
    total = len(pokemon)

    stat_rows = []
    for stat in STATS:
        values = [p[stat] for p in pokemon if p[stat] is not None]
        stat_rows.append((
            stat,
            len(values),
            fmt_num(statistics.fmean(values)) if values else "N/A",
            fmt_num(statistics.stdev(values)) if len(values) > 1 else "N/A",
            min(values) if values else "N/A",
            max(values) if values else "N/A",
        ))
    print("\n[能力值]")
    print_table(["能力", "有效筆數", "平均", "樣本標準差", "最小值", "最大值"], stat_rows)

    print_distribution(
        "Type 1",
        Counter(p["Type 1"] if not is_blank(p["Type 1"]) else "(缺值)" for p in pokemon),
        total,
    )
    print_distribution(
        "Type 2",
        Counter("無次屬性" if is_blank(p["Type 2"]) else p["Type 2"] for p in pokemon),
        total,
    )
    print_distribution(
        "屬性數",
        Counter("雙屬性" if not is_blank(p["Type 2"]) else "單屬性" for p in pokemon),
        total,
    )
    print_distribution(
        "Generation",
        Counter(p["Generation"] if p["Generation"] is not None else "(缺值)" for p in pokemon),
        total,
        ordered=True,
    )
    print_distribution(
        "Legendary 原始值",
        Counter(p["Legendary"] if p["Legendary"] is not None else "(NULL)" for p in pokemon),
        total,
    )
    print_distribution(
        "Legendary 轉換後",
        Counter(LEGENDARY_LABEL[to_bool(p["Legendary"])] for p in pokemon),
        total,
    )


def print_distribution(title, counter, total, ordered=False):
    items = sorted(counter.items(), key=lambda kv: str(kv[0])) if ordered else counter.most_common()
    print(f"\n[{title}]")
    print_table([title, "筆數", "比例"], [(key, n, fmt_pct(n, total)) for key, n in items])


def compare_stats(valid):
    section("3. 能力差值與勝負")
    rows = []
    for stat in STATS:
        f_key, s_key = f"f_{FIELD[stat]}", f"s_{FIELD[stat]}"
        comparable = tie = diff = higher_wins = excluded = 0
        for rec in valid:
            a, b = rec[f_key], rec[s_key]
            if a is None or b is None:
                excluded += 1
                continue
            comparable += 1
            if a == b:
                tie += 1
                continue
            diff += 1
            if (a > b) == rec["first_won"]:
                higher_wins += 1
        rows.append((
            stat, comparable, tie, diff, higher_wins, fmt_pct(higher_wins, diff), excluded,
        ))
    print_table(
        ["能力", "可比較場次", "同值場次", "不同值場次", "較高者獲勝場次", "較高者勝率", "缺值排除筆數"],
        rows,
    )


def categorical_effects(valid):
    section("4. 類別特徵與勝負")
    sd_n = sd_w = 0
    lg_n = lg_w = lg_unknown = 0
    gen_n = gen_w = gen_excluded = 0
    for rec in valid:
        f_dual = not is_blank(rec["f_type2"])
        s_dual = not is_blank(rec["s_type2"])
        if f_dual != s_dual:
            sd_n += 1
            if f_dual == rec["first_won"]:
                sd_w += 1

        fl, sl = to_bool(rec["f_legendary"]), to_bool(rec["s_legendary"])
        if fl is None or sl is None:
            lg_unknown += 1
        elif fl != sl:
            lg_n += 1
            if fl == rec["first_won"]:
                lg_w += 1

        fg, sg = rec["f_gen"], rec["s_gen"]
        if fg is None or sg is None:
            gen_excluded += 1
        elif fg != sg:
            gen_n += 1
            if (fg > sg) == rec["first_won"]:
                gen_w += 1

    print_table(
        ["比較條件", "場次", "特徵方勝場", "特徵方勝率", "排除筆數"],
        [
            ("一方單屬性、一方雙屬性（特徵方=雙屬性）", sd_n, sd_w, fmt_pct(sd_w, sd_n), "-"),
            ("一方傳說、一方非傳說（特徵方=傳說）", lg_n, lg_w, fmt_pct(lg_w, lg_n), lg_unknown),
            ("雙方世代不同（特徵方=較新世代）", gen_n, gen_w, fmt_pct(gen_w, gen_n), gen_excluded),
        ],
    )


def candidate_features(valid):
    section("5. 候選特徵範例")
    complete = [
        rec for rec in valid
        if all(rec[f"f_{FIELD[s]}"] is not None and rec[f"s_{FIELD[s]}"] is not None for s in STATS)
    ]
    print(f"可計算六項能力差值的有效對戰: {len(complete)} 筆；以下列出前 5 筆（差值 = 第一隻 - 第二隻）")
    rows = []
    for rec in complete[:5]:
        diffs = [rec[f"f_{FIELD[s]}"] - rec[f"s_{FIELD[s]}"] for s in STATS]
        rows.append((
            rec["combat_id"], rec["first_id"], rec["second_id"],
            1 if rec["first_won"] else 0, *diffs,
        ))
    headers = ["對戰列號", "第一隻編號", "第二隻編號", "標籤(第一隻勝=1)", *[f"{s}差" for s in STATS]]
    print_table(headers, rows)


def main():
    db = connect_readonly()
    try:
        total_combats = db.execute("SELECT COUNT(*) FROM combats").fetchone()[0]
        joined = load_joined(db)
        valid = [rec for rec in joined if rec["winner"] in (rec["first_id"], rec["second_id"])]
        check_data(db, total_combats, joined, valid)
        describe_pokemon(db)
        compare_stats(valid)
        categorical_effects(valid)
        candidate_features(valid)
    finally:
        db.close()


if __name__ == "__main__":
    main()
