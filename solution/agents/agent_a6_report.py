# -*- coding: utf-8 -*-
"""
Агент A6 (задание 1.6) — Подготовка отчёта по модулю А.
Формирует Markdown-отчёт: функционал агентов, обоснование подходов,
примеры контента, структура БД, форматы файлов, примеры записей датасета.
Запуск: python agent_a6_report.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATASET_DIR, DB_PATH, REPORTS_DIR, get_conn

REQ_NAMES = ["Наличие аннотации", "Наличие темы/названия",
             "Достаточный объем текста (>2000 зн.)",
             "Структурность (заголовки/абзацы)",
             "Наличие иллюстративного материала", "Описание медиакомпонентов",
             "Тематическая связность (термины повторяются)",
             "Отсутствие запрещенного контента",
             "Наличие ключевых слов (>=5 уникальных)",
             "Завершенность (есть выводы)"]


def main():
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) c FROM materials").fetchone()["c"]
    ng = conn.execute("SELECT COUNT(*) c FROM materials WHERE is_generated=1").fetchone()["c"]
    npass = conn.execute("SELECT COUNT(*) c FROM materials WHERE moderation_passed=1").fetchone()["c"]
    feats = json.loads((REPORTS_DIR / "features_report.json").read_text(encoding="utf-8")) \
        if (REPORTS_DIR / "features_report.json").exists() else {}
    sample = conn.execute("""SELECT uid, domain, topic, lesson_type, difficulty,
        moderation_score, is_generated, has_prev, has_next,
        LENGTH(text) tl FROM materials ORDER BY RANDOM() LIMIT 3""").fetchall()
    gen = conn.execute("""SELECT uid, topic, moderation_score FROM materials
                          WHERE is_generated=1 LIMIT 3""").fetchall()

    # выполнение требований по категориям
    req_stats = {r: 0 for r in REQ_NAMES}
    for row in conn.execute("SELECT moderation_detail FROM materials"):
        try:
            d = json.loads(row["moderation_detail"] or "{}")
            for k in req_stats:
                req_stats[k] += bool(d.get(k))
        except Exception:
            pass

    lines = [
        "# Отчёт по Модулю А. Сбор и предварительная обработка данных",
        "",
        "## 1. Реализованный функционал (агенты модуля)",
        "| Агент | Задание | Функция | Выход |",
        "|---|---|---|---|",
        "| A1 `agent_a1_loader.py` | 1.1 | Загрузка материалов по источникам, аналитические справки, модерация, upsert в SQLite без дублирования | таблица `materials`, `sources.json` |",
        "| A2 `agent_a2_dataset.py` | 1.2 | Анализ изображений/таблиц, описание медиа, вывод о допустимости | колонки `media_desc`, `is_admissible`, `dataset/dataset.csv` |",
        "| A3 `agent_a3_features.py` | 1.3 | Корреляционный анализ + permutation importance; признаки has_prev/has_next | `reports/features_report.json` |",
        "| A4 `agent_a4_structure.py` | 1.4 | Описание атрибутов, уникальности, частоты, длины текстов; сводки и графики | `structure_summary.csv`, `figures/dataset_structure.png` |",
        "| A5 `agent_a5_augment.py` | 1.5 | Генерация недостающих «мостов» между темами, модерация, добавление в БД/датасет | материалы `is_generated=1` |",
        "| A6 (этот отчёт) | 1.6 | Сводный отчёт модуля | `reports/module_A_report.md` |",
        "",
        "## 2. Обоснование выбранных подходов",
        "- **Хранение**: SQLite — не требует сервера, работает офлайн, поддерживает "
        "идемпотентный upsert по уникальному ключу `uid` (повторный запуск A1 "
        "обновляет записи без дублирования).",
        "- **Модерация**: детерминированный чек-лист из 10 требований методических "
        "рекомендаций (аннотация, объём, структурность, медиа, связность, "
        "безопасность контента, завершённость); порог прохода — 70%.",
        "- **Схожесть материалов**: TF-IDF (собственная лёгкая реализация) + косинусная "
        "мера; пороги связности задаются перцентилями распределения близостей, что "
        "адаптивно для каждой дисциплины.",
        "- **Значимость признаков**: корреляция Пирсона с целевыми метриками и "
        "permutation importance на RandomForest — два взаимодополняющих метода.",
        "- **Генерация материалов**: компилятивный шаблонный генератор на терминах "
        "исходных текстов (полностью офлайн, без внешних LLM), обязательная "
        "проверка агентом-модератором A1 перед добавлением.",
        "",
        "## 3. Структура базы данных",
        "```sql",
        ]
    schema = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name IN "
        "('materials','labels','model_versions','agent_log')").fetchall()
    lines += [s["sql"] + ";" for s in schema]
    lines += [
        "```",
        "",
        f"## 4. Итоговые показатели",
        f"- Всего материалов в БД: **{n}** (исходных: {n - ng}, сгенерировано агентом A5: **{ng}**)",
        f"- Прошли модерацию: **{npass}** ({100 * npass / n:.1f}%)",
        f"- Выполнение требований методических рекомендаций: " +
        ", ".join(f"«{k}» {100 * v / n:.0f}%" for k, v in list(req_stats.items())[:5]) + " …",
        "",
        "### Топ значимых признаков схожести (permutation importance)",
        *[f"- {f}: {feats.get('перmutation_importance', feats.get('permutation_importance', {})).get(f, '-')}"
          for f in feats.get("топ_признаки", [])],
        "",
        "## 5. Примеры записей датасета",
        "| uid | предмет | тема | тип занятия | сложность | модерация % | ген. | prev | next | длина |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in sample:
        lines.append(f"| {s['uid']} | {s['domain']} | {s['topic'][:40]}… | "
                     f"{s['lesson_type']} | {s['difficulty']} | {s['moderation_score']} | "
                     f"{s['is_generated']} | {s['has_prev']} | {s['has_next']} | {s['tl']} |")
    lines += ["", "### Примеры сгенерированных материалов (агент A5)",
              "| uid | тема | модерация % |", "|---|---|---|"]
    for g in gen:
        lines.append(f"| {g['uid']} | {g['topic'][:60]} | {g['moderation_score']} |")
    lines += [
        "",
        "## 6. Форматы итоговых файлов",
        "- `db/materials.db` — SQLite (таблицы materials, labels, model_versions, agent_log)",
        "- `dataset/dataset.csv` — CSV, разделитель «,», UTF-8",
        "- `dataset/generated_examples.json` — JSON-список сгенерированных материалов",
        "- `output/reports/features_report.json`, `dataset_structure.json`, "
        "`structure_summary.csv` — отчёты 1.3–1.4",
        "- `output/figures/dataset_structure.png` — визуализации распределений",
    ]
    out = REPORTS_DIR / "module_A_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("Отчёт сохранён:", out)
    conn.close()


if __name__ == "__main__":
    main()
