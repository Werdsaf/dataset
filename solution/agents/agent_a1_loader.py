# -*- coding: utf-8 -*-
"""
Агент A1 (задание 1.1) — Загрузка данных и модерация.

Загружает учебные материалы по списку источников (локальные пути/ссылки,
файл sources.json), формирует аналитическую справку:
{id, предмет, тема, текст, аннотация}, проводит модерацию по методическим
требованиям и сохраняет в SQLite. Повторный запуск обновляет записи без
дублирования (upsert по uid).

Запуск:  python agent_a1_loader.py [--sources path/to/sources.json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DOMAINS, SOURCE_DATASET, get_conn, init_db, log_event,
                    moderate, tokenize)

DEFAULT_SOURCES = Path(__file__).resolve().parent / "sources.json"


def build_sources_index():
    """Формирует список «ссылок» на материалы исходного корпуса.
    В конкурсном задании ссылки указываются в источниках; здесь корпус
    представлен локальными путями к папкам материалов."""
    srcs = []
    for folder, domain in DOMAINS.items():
        d = SOURCE_DATASET / folder
        if not d.exists():
            continue
        for mat in sorted(d.iterdir()):
            if mat.is_dir() and (mat / "text.txt").exists():
                srcs.append({"url": str(mat), "domain": domain})
    return srcs


def load_material(url: str, domain: str) -> dict:
    p = Path(url)
    topic = (p / "name.txt").read_text(encoding="utf-8", errors="ignore").strip() \
        if (p / "name.txt").exists() else p.name
    abstract = (p / "abstract.txt").read_text(encoding="utf-8", errors="ignore").strip() \
        if (p / "abstract.txt").exists() else ""
    text = (p / "text.txt").read_text(encoding="utf-8", errors="ignore").strip()

    # оценка сложности: доля длинных слов + плотность терминов
    toks = tokenize(text[:20000]) or [""]
    long_share = sum(1 for t in toks if len(t) > 9) / len(toks)
    uniq_share = len(set(toks)) / len(toks)
    difficulty = round(min(5, max(1, 1 + 4 * (0.6 * long_share + 0.4 * uniq_share) * 2)), 2)

    # тип занятия по признакам текста
    tl = text.lower()
    if "эксперимент" in tl or "исследовани" in tl and "метод" in tl:
        lesson_type = "Лабораторное занятие"
    elif "практическ" in tl and ("задач" in tl or "упражнен" in tl):
        lesson_type = "Практическое занятие"
    elif "обзор" in tl or "введени" in tl or "основы" in tl:
        lesson_type = "Лекция"
    else:
        lesson_type = "Самостоятельное изучение"

    n_images = len(list(p.glob("image_*.png")))
    n_tables = len(list(p.glob("table_*.png")))

    return {
        "uid": f"{domain}:{p.name}",
        "source_url": url,
        "domain": domain,
        "topic": topic,
        "text": text,
        "abstract": abstract,
        "lesson_type": lesson_type,
        "difficulty": difficulty,
        "n_images": n_images,
        "n_tables": n_tables,
        "media_desc": "",  # заполняет агент A2
    }


def upsert(conn, m, mod):
    conn.execute("""
        INSERT INTO materials(uid, source_url, domain, topic, text, abstract,
            lesson_type, difficulty, n_images, n_tables, media_desc,
            moderation_passed, moderation_score, moderation_detail, is_generated,
            updated_at)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,datetime('now'))
        ON CONFLICT(uid) DO UPDATE SET
            topic=excluded.topic, text=excluded.text, abstract=excluded.abstract,
            lesson_type=excluded.lesson_type, difficulty=excluded.difficulty,
            n_images=excluded.n_images, n_tables=excluded.n_tables,
            moderation_passed=excluded.moderation_passed,
            moderation_score=excluded.moderation_score,
            moderation_detail=excluded.moderation_detail,
            updated_at=datetime('now')
    """, (m["uid"], m["source_url"], m["domain"], m["topic"], m["text"],
          m["abstract"], m["lesson_type"], m["difficulty"], m["n_images"],
          m["n_tables"], m["media_desc"], mod["passed"], mod["score"],
          json.dumps(mod["detail"], ensure_ascii=False)))


def main(sources_path: Path):
    conn = init_db()
    if sources_path.exists():
        sources = json.loads(sources_path.read_text(encoding="utf-8"))
    else:
        sources = build_sources_index()
        sources_path.write_text(json.dumps(sources, ensure_ascii=False, indent=1),
                                encoding="utf-8")
        print(f"Сформирован список источников: {sources_path} ({len(sources)})")

    added = updated = rejected = 0
    for s in sources:
        try:
            m = load_material(s["url"], s["domain"])
        except Exception as e:
            print(f"!! ошибка загрузки {s['url']}: {e}")
            continue
        mod = moderate(dict(m))
        exists = conn.execute("SELECT id FROM materials WHERE uid=?",
                              (m["uid"],)).fetchone()
        upsert(conn, m, mod)
        if exists:
            updated += 1
        else:
            added += 1
        if not mod["passed"]:
            rejected += 1
    conn.commit()
    log_event(conn, "A1_loader", "ingest",
              f"added={added}, updated={updated}, rejected={rejected}")
    total = conn.execute("SELECT COUNT(*) c FROM materials").fetchone()["c"]
    print(f"Загрузка завершена: новых={added}, обновлено={updated}, "
          f"не прошло модерацию={rejected}. Всего в БД: {total}")
    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=str(DEFAULT_SOURCES))
    args = ap.parse_args()
    main(Path(args.sources))
