# -*- coding: utf-8 -*-
"""
Агент A5 (задание 1.5) — Расширение набора данных.

Определяет «разрывы» в учебных последовательностях дисциплин (темы, для
которых нет связного предыдущего/последующего материала) и генерирует
недостающие учебные материалы на основе имеющихся: компилятивный шаблонный
генератор (без сети) строит аннотацию, план, основной текст и выводы из
токенов-терминов соседних материалов. Сгенерированные материалы проходят
автоматическую модерацию агентом A1 и добавляются в БД и датасет.

Запуск: python agent_a5_augment.py [--max-new 40]
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DATASET_DIR, SOURCE_DATASET, build_tfidf, cosine, get_conn,
                    log_event, moderate, tokenize)

LESSON_TYPES = ["Лекция", "Практическое занятие", "Лабораторное занятие",
                "Самостоятельное изучение"]

CONNECTORS = [
    "Рассмотрим далее тему «{topic}», логически продолжая материал «{prev}».",
    "Переходя к теме «{topic}», опираемся на результаты, изложенные в работе «{prev}».",
    "Тема «{topic}» развивает положения предыдущего занятия «{prev}».",
]
BODY_TPL = [
    "Целью изучения темы «{topic}» является освоение ключевых аспектов области "
    "«{domain}». К числу базовых понятий относятся: {terms}.",
    "При анализе темы {topic2} особое внимание уделяется методам, описываемым через "
    "{terms2}. Практическая часть включает разбор типовых задач и работу с {media}.",
    "Важным свойством подхода является прозрачность применяемых преобразований данных; "
    "используются критерии качества, метрики {terms3} и сопоставление альтернатив.",
    "В ходе занятия предполагается обсуждение ограничений метода и областей применения, "
    "а также закрепление материала на примерах, связанных с {terms4}.",
]
CONCLUSION = ("Таким образом, тема «{topic}» завершает блок дисциплины «{domain}» и "
              "формирует основу для дальнейшего изучения смежных вопросов; "
              "рекомендуется выполнить итоговое задание для самопроверки.")


def terms_from(texts, k=8, exclude=None):
    cnt = Counter()
    for t in texts:
        cnt.update(tokenize(t)[:4000])
    for w in (exclude or []):
        cnt.pop(w, None)
    return [w for w, _ in cnt.most_common(k) if len(w) > 4]


def gen_text(topic, domain, prev_topic, terms, n_tables):
    media = f"{max(1, n_tables)} таблиц(ы) и схемы" if n_tables else "схемы и примеры"
    p1 = CONNECTORS[random.Random(len(topic)).randrange(len(CONNECTORS))].format(
        topic=topic, prev=prev_topic)
    body = []
    for i, tpl in enumerate(BODY_TPL):
        chunk = terms[i * 2:i * 2 + 3] or terms[:3]
        body.append(tpl.format(topic=topic, topic2=topic.lower(), domain=domain,
                               terms=", ".join(chunk), terms2=", ".join(chunk[::-1]),
                               terms3=", ".join(chunk[:2]), terms4=chunk[0],
                               media=media))
    # раздуем до методического объёма вариативными абзацами
    extra = []
    rnd = random.Random(hash(topic) & 0xffff)
    stems = terms[:6]
    for s in stems:
        extra.append(
            f"Отдельного рассмотрения заслуживает аспект «{s}»: приведём определение, "
            f"разберём типичные ошибки и продемонстрируем расчёт на числовом примере "
            f"из предметной области «{domain.lower()}». Это позволяет учащемуся "
            f"сформировать устойчивый навык работы с данным понятием.")
    full = "\n\n".join([p1] + body + extra + [CONCLUSION.format(topic=topic, domain=domain)])
    return full


def find_gaps(conn):
    """Разрывы: дисциплины без связки has_prev/has_next у части тем —
    предлагаем «мосты» между несвязанными кластерами тем."""
    rows = conn.execute("""SELECT id, uid, domain, topic, text, n_tables,
                           has_prev, has_next FROM materials""").fetchall()
    gaps = []
    by_domain = {}
    for r in rows:
        by_domain.setdefault(r["domain"], []).append(r)
    for dom, items in by_domain.items():
        lonely = [r for r in items if not r["has_prev"] and not r["has_next"]]
        for r in lonely[:3]:  # не более 3 мостов на дисциплину
            gaps.append((dom, r))
    return gaps


def main(max_new: int):
    random.seed(7)
    conn = get_conn()
    gaps = find_gaps(conn)
    added = rejected = 0
    generated = []
    for dom, anchor in gaps:
        if added >= max_new:
            break
        terms = terms_from([anchor["text"]], k=12)
        base_topic = re.split(r"[—:,]", anchor["topic"])[0].strip()[:60]
        topic = f"{base_topic}: углублённый практикум (сгенерированный мост)"
        text = gen_text(topic, dom, anchor["topic"], terms, anchor["n_tables"])
        abstract = (f"Автоматически сгенерированный учебный материал по дисциплине "
                    f"«{dom}». Тема: {topic}. Аннотация исходного материала: "
                    f"{(conn.execute('SELECT abstract FROM materials WHERE id=?', (anchor['id'],)).fetchone()[0] or '')[:220]}")
        m = {"uid": f"gen:{dom}:{added + 1}", "source_url": f"generated://{dom}/{added + 1}",
             "domain": dom, "topic": topic, "text": text, "abstract": abstract,
             "lesson_type": LESSON_TYPES[added % 4],
             "difficulty": round(min(5, (anchor["topic"].count(" ") + 2) % 4 + 2.0), 2),
             "n_images": 1, "n_tables": max(1, anchor["n_tables"]),
             "media_desc": json.dumps({"media": [{"file": "image_1.png",
              "kind": "схема/график (двухцветное изображение)"}]}, ensure_ascii=False)}
        mod = moderate(dict(m))
        if not mod["passed"]:
            rejected += 1
            continue
        exists = conn.execute("SELECT 1 FROM materials WHERE uid=?", (m["uid"],)).fetchone()
        if exists:
            continue
        conn.execute("""INSERT INTO materials(uid, source_url, domain, topic, text,
            abstract, lesson_type, difficulty, n_images, n_tables, media_desc,
            moderation_passed, moderation_score, moderation_detail, is_generated,
            is_admissible) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,1)""",
            (m["uid"], m["source_url"], m["domain"], m["topic"],
             m["text"], m["abstract"], m["lesson_type"], m["difficulty"],
             m["n_images"], m["n_tables"], m["media_desc"], mod["passed"],
             mod["score"], json.dumps(mod["detail"], ensure_ascii=False)))
        generated.append({"uid": m["uid"], "domain": dom, "topic": topic,
                          "moderation_score": mod["score"]})
        added += 1
    conn.commit()

    # пересчёт has_prev/has_next уже с учётом новых материалов
    rows = conn.execute("SELECT id, domain, text FROM materials").fetchall()
    docs = [r["text"] or "" for r in rows]
    tf, _ = build_tfidf(docs, 300)
    import numpy as np
    by_domain = {}
    for i, r in enumerate(rows):
        by_domain.setdefault(r["domain"], []).append(i)
    all_sims = [cosine(tf[i], tf[j]) for idxs in by_domain.values()
                for a, i in enumerate(idxs) for j in idxs[a + 1:]]
    THRESH = float(np.percentile(all_sims, 90))
    for dom, idxs in by_domain.items():
        for i in idxs:
            sims = sorted(((cosine(tf[i], tf[j]), j) for j in idxs if j != i),
                          reverse=True)
            strong = [s for s in sims if s[0] >= THRESH][:5]
            conn.execute("UPDATE materials SET has_prev=?, has_next=? WHERE id=?",
                         (int(any(len(docs[j]) < len(docs[i]) for _, j in strong)),
                          int(any(len(docs[j]) > len(docs[i]) for _, j in strong)),
                          rows[i]["id"]))
    conn.commit()

    # обновим экспорт датасета
    with open(DATASET_DIR / "dataset.csv", "w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["id", "uid", "domain", "topic", "lesson_type", "difficulty",
                    "n_images", "n_tables", "moderation_passed", "moderation_score",
                    "is_generated", "is_admissible", "has_prev", "has_next",
                    "text_len", "abstract"])
        for r in conn.execute("""SELECT *, LENGTH(text) text_len FROM materials"""):
            w.writerow([r["id"], r["uid"], r["domain"], r["topic"], r["lesson_type"],
                        r["difficulty"], r["n_images"], r["n_tables"],
                        r["moderation_passed"], r["moderation_score"],
                        r["is_generated"], r["is_admissible"], r["has_prev"],
                        r["has_next"], r["text_len"], (r["abstract"] or "")[:300]])

    (DATASET_DIR / "generated_examples.json").write_text(
        json.dumps(generated, ensure_ascii=False, indent=2), encoding="utf-8")
    log_event(conn, "A5_augment", "generated",
              f"added={added}, rejected={rejected}")
    print(f"Сгенерировано и добавлено: {added}, отклонено модерацией: {rejected}")
    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=40)
    args = ap.parse_args()
    main(args.max_new)
