# -*- coding: utf-8 -*-
"""
Агент A4 (задание 1.4) — Описание структуры набора данных.

Для каждого атрибута готовит текстовое описание (расшифровка, назначение,
тип данных), считает количество уникальных значений, частоты и наиболее
распространённые значения; для текстовых атрибутов — среднюю/мин/макс длину
символов и слов. Результаты: сводные таблицы CSV + визуализации PNG.

Запуск: python agent_a4_structure.py
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIGURES_DIR, REPORTS_DIR, get_conn, log_event, tokenize

DESCRIBE = {
    "uid": ("str", "Уникальный идентификатор материала (домен:папка)",
            "Ключ upsert, защита от дублирования"),
    "domain": ("str", "Предмет (дисциплина)", "Группировка, покрытие тем"),
    "topic": ("str", "Тема/название материала", "Семантика, поиск, траектории"),
    "lesson_type": ("str", "Тип занятия", "Распределение по типам занятий"),
    "difficulty": ("float", "Оценка сложности освоения 1–5",
                   "Кластеризация по сложности, прогноз времени"),
    "n_images": ("int", "Число иллюстраций", "Мультимодальность, модерация"),
    "n_tables": ("int", "Число таблиц", "Мультимодальность, модерация"),
    "moderation_passed": ("bool", "Прошёл ли материал модерацию",
                          "Фильтр качества датасета"),
    "moderation_score": ("float", "% выполнения методических требований",
                         "Аналитика покрытия требований"),
    "is_generated": ("bool", "Сгенерирован ли материал агентом",
                     "Доля синтетики в датасете"),
    "is_admissible": ("bool", "Допустимость использования материала",
                      "Итоговый вывод агента формирования датасета"),
    "has_prev": ("bool", "Есть предыдущий материал в последовательности",
                 "Планирование траектории"),
    "has_next": ("bool", "Есть последующий материал в последовательности",
                 "Планирование траектории"),
    "text_len": ("int", "Длина текста (симв.)", "Характеристика контента"),
    "abstract": ("str", "Аннотация", "Краткое содержание"),
}


def main():
    conn = get_conn()
    rows = conn.execute("""SELECT uid, domain, topic, lesson_type, difficulty,
        n_images, n_tables, moderation_passed, moderation_score, is_generated,
        is_admissible, has_prev, has_next, LENGTH(text) text_len, abstract,
        text FROM materials""").fetchall()

    # ---- сводная таблица по категориальным/бинарным признакам ----------
    cat_stats = {}
    for col in ["domain", "lesson_type", "moderation_passed", "is_generated",
                "has_prev", "has_next"]:
        vals = [r[col] for r in rows]
        cnt = Counter(vals)
        cat_stats[col] = {"unique": len(cnt),
                          "top": cnt.most_common(5),
                          "freqs": dict(cnt)}

    num_stats = {}
    for col in ["difficulty", "moderation_score", "n_images", "n_tables",
                "text_len"]:
        vals = [float(r[col] or 0) for r in rows]
        cnt = Counter(vals)
        num_stats[col] = {
            "unique": len(cnt), "mean": round(sum(vals) / len(vals), 2),
            "min": min(vals), "max": max(vals), "top": cnt.most_common(3)}

    # ---- характеристики текстовых полей --------------------------------
    text_stats = {}
    for col in ["topic", "abstract", "text"]:
        lens = [len(r[col] or "") for r in rows]
        words = [len(tokenize(r[col] or "")) for r in rows]
        text_stats[col] = {
            "avg_chars": round(sum(lens) / len(lens), 1),
            "min_chars": min(lens), "max_chars": max(lens),
            "avg_words": round(sum(words) / len(words), 1),
            "min_words": min(words), "max_words": max(words)}

    # ---- сохранение сводных таблиц --------------------------------------
    with open(REPORTS_DIR / "structure_summary.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["атрибут", "тип", "описание", "назначение", "уникальных",
                    "частые_значения"])
        for col, (typ, desc, use) in DESCRIBE.items():
            st = cat_stats.get(col) or num_stats.get(col) or {}
            top = "; ".join(f"{k}:{v}" for k, v in (st.get("top") or [])[:3])
            w.writerow([col, typ, desc, use, st.get("unique", ""), top])
        for col, ts in text_stats.items():
            w.writerow([col, "text", f"длины: avg {ts['avg_chars']} зн., "
                        f"{ts['avg_words']} сл.", "характеристики текста", "", ""])

    # ---- визуализации ----------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    d = Counter(r["domain"] for r in rows)
    axes[0, 0].bar(d.keys(), d.values(), color="#4c72b0")
    axes[0, 0].set_title("Материалы по дисциплинам")
    axes[0, 0].tick_params(axis="x", rotation=45)

    lt = Counter(r["lesson_type"] for r in rows)
    axes[0, 1].pie(lt.values(), labels=lt.keys(), autopct="%1.0f%%",
                   colors=["#55a868", "#c44e52", "#8172b2", "#ccb974"])
    axes[0, 1].set_title("Распределение по типам занятий")

    sc = [r["moderation_score"] for r in rows]
    axes[1, 0].hist(sc, bins=18, color="#4c72b0", edgecolor="white")
    axes[1, 0].axvline(70, color="red", ls="--", label="порог модерации")
    axes[1, 0].legend(); axes[1, 0].set_title("Распределение баллов модерации")

    tl = [r["text_len"] / 1000 for r in rows]
    axes[1, 1].hist(tl, bins=25, color="#55a868", edgecolor="white")
    axes[1, 1].set_title("Длина текста, тыс. знаков")
    fig.suptitle("Структура и распределения набора данных")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dataset_structure.png", dpi=110)
    plt.close(fig)

    import json
    (REPORTS_DIR / "dataset_structure.json").write_text(json.dumps({
        "описание_атрибутов": {k: {"тип": v[0], "расшифровка": v[1],
                                   "назначение": v[2]} for k, v in DESCRIBE.items()},
        "категориальные": {k: {"unique": v["unique"],
                               "top": [[str(a), b] for a, b in v["top"]]}
                           for k, v in cat_stats.items()},
        "числовые": num_stats, "текстовые": text_stats},
        ensure_ascii=False, indent=2), encoding="utf-8")

    log_event(conn, "A4_structure", "described", f"attrs={len(DESCRIBE)}")
    print("Структура описана:", REPORTS_DIR / "dataset_structure.json")
    conn.close()


if __name__ == "__main__":
    main()
