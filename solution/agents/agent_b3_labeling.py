# -*- coding: utf-8 -*-
"""
Агент B3/B4 (задания 2.3–2.4) — Разметка и оценка качества кластеризации.

Размечает набор данных тремя системами меток:
 1) parallel_cluster   — группы материалов для СОВМЕСТНОГО (параллельного)
    изучения: темы из независимых последовательностей (низкая TF-IDF
    связность между материалами внутри группы);
 2) sequence_cluster   — группы для ПОСЛЕДОВАТЕЛЬНОГО изучения: высокая
    тематическая связность + упорядоченность по объёму/сложности;
 3) difficulty_cluster — группы по сложности освоения.

Сравниваются методы KMeans / Agglomerative / DBSCAN на TF-IDF признаках,
метрики: silhouette, calinski-harabasz, davies-bouldin. Лучшие метки
сохраняются в таблицу labels (основа обучения модуля В).

Запуск: python agent_b3_labeling.py
"""
import json
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIGURES_DIR, REPORTS_DIR, build_tfidf, get_conn, log_event

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score)
from sklearn.neighbors import NearestNeighbors


def cluster_metrics(X, labels):
    if len(set(labels)) < 2:
        return None
    return {
        "n_clusters": int(len(set(labels))),
        "silhouette": round(float(silhouette_score(X, labels)), 3),
        "calinski": round(float(calinski_harabasz_score(X, labels)), 1),
        "davies": round(float(davies_bouldin_score(X, labels)), 3),
    }


def run_methods(Xred, k):
    results = {}
    for name, model in [
        ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=42)),
        ("agglomerative", AgglomerativeClustering(n_clusters=k)),
        ("dbscan", DBSCAN(eps=float(np.percentile(
            NearestNeighbors(n_neighbors=5).fit(Xred).kneighbors()[0][:, -1], 90)),
            min_samples=5)),
    ]:
        labels = model.fit_predict(Xred)
        mres = cluster_metrics(Xred, labels)
        if mres:
            results[name] = {"labels": labels.tolist(), **mres}
    return results


def main():
    conn = get_conn()
    rows = conn.execute("SELECT id, domain, text, has_prev, has_next FROM materials").fetchall()
    ids = [r["id"] for r in rows]
    docs = [r["text"] or "" for r in rows]

    tf, vocab = build_tfidf(docs, max_features=250)
    X = np.array(tf)
    Xred = PCA(n_components=20, random_state=42).fit_transform(X)

    # ---------- 1) параллельное изучение: междисциплинная независимость ----
    # признаки: TF-IDF минус «последовательная» компонента; материалы из
    # разных дисциплин априори независимы -> кластеризуем междоменную близость
    par = AgglomerativeClustering(n_clusters=8, linkage="average").fit_predict(Xred)
    # очистка: внутри кластера не должно быть сильно связанных пар
    sim_matrix = X @ X.T
    conflicts = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if par[i] == par[j] and rows[i]["domain"] == rows[j]["domain"] \
                    and sim_matrix[i, j] > 0.55:
                par[j] = par[i] + 100  # выносим конфликтующий в отдельную группу
                conflicts += 1
    par = np.unique(par, return_inverse=True)[1]

    # ---------- 2) последовательное изучение --------------------------------
    # сильные тематические связи + порядок difficulty/объёма
    seq_feats = np.column_stack([
        Xred[:, :12],
        np.array([[len(d), r["has_prev"], r["has_next"]]
                  for d, r in zip(docs, rows)], dtype=float)])
    seq_res = run_methods(seq_feats, 6)
    best_seq = max(seq_res.items(), key=lambda kv: kv[1]["silhouette"])
    seq = np.array(best_seq[1]["labels"])

    # ---------- 3) сложность ------------------------------------------------
    diff_vals = np.array([[r["difficulty"] or 3.0] for r in conn.execute(
        "SELECT id, difficulty FROM materials ORDER BY id")])
    diff_ids = [r["id"] for r in conn.execute(
        "SELECT id FROM materials ORDER BY id")]
    id2diff = dict(zip(diff_ids, diff_vals.ravel()))
    Xd = np.column_stack([Xred[:, :6], [[id2diff.get(i, 3)] for i in ids]])
    dif_res = run_methods(Xd, 3)
    best_dif = max(dif_res.items(), key=lambda kv: kv[1]["silhouette"])
    dif = np.array(best_dif[1]["labels"])

    # ---------- сравнение методов (задание 2.4) ----------------------------
    par_res = run_methods(Xred, 8)
    comparison = {"parallel": par_res, "sequence": seq_res, "difficulty": dif_res}
    chosen = {t: max(res.items(), key=lambda kv: kv[1]["silhouette"])[0]
              for t, res in comparison.items()}

    # ---------- сохранение меток -------------------------------------------
    conn.execute("DELETE FROM labels")
    for mid, p, s, dv in zip(ids, par, seq, dif):
        conn.execute("""INSERT INTO labels(material_id, parallel_cluster,
                        sequence_cluster, difficulty_cluster) VALUES(?,?,?,?)""",
                     (int(mid), int(p), int(s), int(dv)))
    conn.commit()

    # ---------- визуализация PCA -------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    X2 = PCA(n_components=2, random_state=42).fit_transform(Xred)
    for ax, lab, title in [(axes[0], par, "Параллельное изучение"),
                           (axes[1], seq, f"Последовательное ({best_seq[0]})"),
                           (axes[2], dif, f"Сложность ({best_dif[0]})")]:
        sc = ax.scatter(X2[:, 0], X2[:, 1], c=lab, cmap="tab10", s=12, alpha=.75)
        ax.set_title(title)
    fig.colorbar(sc, ax=axes, shrink=.8)
    fig.suptitle("Кластеры разметки (PCA 2D пространства TF-IDF)")
    fig.savefig(FIGURES_DIR / "clusters.png", dpi=110)
    plt.close(fig)

    summary = {
        "выбранные_методы": chosen,
        "метрики": {t: {m: {k: v for k, v in r.items() if k != "labels"}
                        for m, r in res.items()} for t, res in comparison.items()},
        "конфликты_параллельных_кластеров_исправлено": conflicts,
    }
    (REPORTS_DIR / "labeling_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log_event(conn, "B3_labeling", "labeled", json.dumps(chosen, ensure_ascii=False))
    print("Выбранные методы:", chosen)
    conn.close()


if __name__ == "__main__":
    main()
