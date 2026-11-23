# -*- coding: utf-8 -*-
"""
Агент A3 (задание 1.3) — Предобработка и выделение значимых атрибутов.

Анализирует атрибуты, влияющие на схожесть учебных материалов:
 - корреляционный анализ (Пирсон/Спирмен) числовых признаков с целевыми
   показателями (сложность, модерация);
 - permutation importance на RandomForest (sklearn) для признака схожести;
 - дополняет датасет флагами has_prev / has_next — наличие предыдущего или
   последующего материала в учебной последовательности по тематике
   (связь определяется TF-IDF близостью внутри дисциплины).

Результаты: output/reports/features_report.json + колонки БД.
Запуск: python agent_a3_features.py
"""
import json
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIGURES_DIR, REPORTS_DIR, build_tfidf, cosine, get_conn, \
    log_event, tokenize

FEATURES = ["difficulty", "text_len", "n_images", "n_tables", "topic_len",
            "abstract_len", "paragraphs", "sentences"]


def extract_matrix(conn):
    rows = conn.execute("""SELECT id, text, topic, abstract, difficulty,
                           n_images, n_tables FROM materials""").fetchall()
    X, ids = [], []
    for r in rows:
        t = r["text"] or ""
        X.append([
            r["difficulty"] or 0,
            len(t),
            r["n_images"],
            r["n_tables"],
            len(r["topic"] or ""),
            len(r["abstract"] or ""),
            t.count("\n\n"),
            max(1, len([s for s in t.replace("!", "?").split(".") if len(s) > 40])),
        ])
        ids.append(r["id"])
    return np.array(X, dtype=float), ids


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / d) if d else 0.0


def main():
    conn = get_conn()
    X, ids = extract_matrix(conn)
    y_diff = X[:, 0]
    y_mod = np.array([r["moderation_score"] for r in conn.execute(
        "SELECT moderation_score FROM materials")], dtype=float)

    # --- корреляционный анализ -------------------------------------------
    corr = {f: {"pearson_difficulty": round(pearson(X[:, i], y_diff), 3),
                "pearson_moderation": round(pearson(X[:, i], y_mod), 3)}
            for i, f in enumerate(FEATURES)}

    # --- permutation importance (сквозь sklearn) --------------------------
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import cross_val_score
    mask = ~np.isnan(X).any(axis=1)
    Xm, ym = X[mask], y_mod[mask]
    rf = RandomForestRegressor(n_estimators=120, random_state=42)
    rf.fit(Xm, ym)
    pi = permutation_importance(rf, Xm, ym, n_repeats=15, random_state=42)
    perm_imp = {f: round(float(v), 4) for f, v in zip(FEATURES, pi.importances_mean)}
    ranked = sorted(perm_imp.items(), key=lambda kv: -kv[1])
    significant = [f for f, v in ranked if v > 0.01]

    # --- признаки связности последовательности has_prev / has_next --------
    rows = conn.execute("SELECT id, domain, text FROM materials").fetchall()
    docs = [r["text"] or "" for r in rows]
    tf, vocab = build_tfidf(docs, max_features=300)
    by_domain = {}
    for i, r in enumerate(rows):
        by_domain.setdefault(r["domain"], []).append(i)

    # порог тематической связанности — 90-й перцентиль распределения парных
    # косинусных близостей внутри дисциплин (сильно связанные пары)
    all_sims = [cosine(tf[i], tf[j])
                for idxs in by_domain.values()
                for a, i in enumerate(idxs) for j in idxs[a + 1:]]
    THRESH = float(np.percentile(all_sims, 90)) if all_sims else 0.5

    prev_cnt = next_cnt = 0
    for dom, idxs in by_domain.items():
        for i in idxs:
            sims = [(cosine(tf[i], tf[j]), rows[j]["id"], j)
                    for j in idxs if j != i]
            sims.sort(reverse=True)
            if not sims:
                continue
            # «предыдущий»: тематически связанный материал меньшего объёма
            # (фундамент → надстройка); «последующий» — большего объёма.
            strong = [s for s in sims if s[0] >= THRESH][:5]
            best_prev = any(len(docs[s[2]]) < len(docs[i]) for s in strong)
            best_next = any(len(docs[s[2]]) > len(docs[i]) for s in strong)
            conn.execute("UPDATE materials SET has_prev=?, has_next=? WHERE id=?",
                         (int(best_prev), int(best_next), rows[i]["id"]))
            prev_cnt += best_prev
            next_cnt += best_next
    conn.commit()

    report = {
        "корреляция_признаков": corr,
        "permutation_importance": perm_imp,
        "значимые_признаки": significant,
        "топ_признаки": [f for f, _ in ranked[:4]],
        "последовательность": {
            "материалов_с_предыдущим": prev_cnt,
            "материалов_с_последующим": next_cnt,
            "порог_схожести": THRESH,
        },
    }
    (REPORTS_DIR / "features_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log_event(conn, "A3_features", "features_done",
              f"significant={significant}")
    print("Значимые признаки:", significant)
    print(f"has_prev: {prev_cnt}, has_next: {next_cnt}")
    conn.close()


if __name__ == "__main__":
    main()
