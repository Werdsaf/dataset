# -*- coding: utf-8 -*-
"""
Агент A2 (задание 1.2) — Формирование набора данных.

Анализирует нетекстовые составляющие учебных материалов (иллюстрации,
атомарные изображения, таблицы как видео/графические фрагменты), формирует
их текстовые описания на основе анализа изображений (размер, соотношение
сторон, цветовая сложность, бинарность — признаки «график/фото/схема»),
дополняет текст материала и выносит итоговый вывод о допустимости
использования материала. Результат — расширенный датасет
output/dataset/dataset.csv + обновление БД.

Запуск: python agent_a2_dataset.py
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATASET_DIR, get_conn, log_event, tokenize

try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False


def describe_image(path: Path) -> dict:
    """Эвристика анализа изображения без сетевых моделей."""
    info = {"file": path.name, "kind": "изображение"}
    if not HAS_PIL:
        info["size_kb"] = round(path.stat().st_size / 1024, 1)
        return info
    try:
        with Image.open(path) as im:
            w, h = im.size
            small = im.convert("RGB").resize((64, 64))
            px = list(small.getdata())
            uniq = len(set(px))
            gray = small.convert("L")
            hist = gray.histogram()
            nz = sum(1 for v in hist if v > 0)
            # доля почти-чёрных и почти-белых пикселей — признак схемы/графика
            ink = sum(hist[:40]) / len(list(gray.getdata()))
            paper = sum(hist[235:]) / len(list(gray.getdata()))
        ratio = w / max(h, 1)
        info.update(width=w, height=h, unique_colors=uniq)
        if uniq < 400 and (ink + paper) > 0.75:
            kind = "схема/график (двухцветное изображение)"
        elif ratio > 2.2:
            kind = "широкоформатная иллюстрация (вероятно, временной ряд/схема)"
        elif uniq > 3000:
            kind = "фотография или цветная диаграмма"
        else:
            kind = "иллюстрация"
        info["kind"] = kind
        info["aspect"] = round(ratio, 2)
    except Exception as e:
        info["error"] = str(e)
    return info


def caption_from_json(folder: Path, name: str) -> str:
    fp = folder / name
    if fp.exists():
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            caps = []
            for item in data if isinstance(data, list) else [data]:
                if isinstance(item, dict):
                    c = item.get("caption") or item.get("description") or \
                        item.get("title") or ""
                    if c:
                        caps.append(str(c)[:200])
            return "; ".join(caps[:6])
        except Exception:
            pass
    return ""


def build_media_desc(source_url: str) -> str:
    p = Path(source_url)
    if not p.exists():
        return ""
    items = []
    for img in sorted(p.glob("image_*.png")):
        d = describe_image(img)
        items.append(d)
    for tbl in sorted(p.glob("table_*.png")):
        d = describe_image(tbl)
        d["kind"] = "таблица (растровое представление): " + d.get("kind", "")
        items.append(d)
    fig_caps = caption_from_json(p, "figures.json")
    tab_caps = caption_from_json(p, "tables.json")
    out = {"media": items, "figure_captions": fig_caps, "table_captions": tab_caps}
    return json.dumps(out, ensure_ascii=False)


def admissibility(row, media_json: str) -> int:
    """Итоговый вывод о допустимости использования материала в целом:
    модерация пройдена И есть хотя бы базовые компоненты."""
    try:
        media = json.loads(media_json) if media_json else {}
    except Exception:
        media = {}
    has_media_meta = bool(media.get("media")) or (row["n_images"] + row["n_tables"] == 0)
    return int(bool(row["moderation_passed"]) and has_media_meta
               and len(row["text"] or "") > 500)


def main():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM materials").fetchall()
    n_media = 0
    for r in rows:
        if not r["media_desc"]:
            md = build_media_desc(r["source_url"])
            if md:
                conn.execute("UPDATE materials SET media_desc=?, updated_at=datetime('now') "
                             "WHERE id=?", (md, r["id"]))
                n_media += 1
        adm = admissibility(r, r["media_desc"] or build_media_desc(r["source_url"]))
        conn.execute("UPDATE materials SET is_admissible=? WHERE id=?", (adm, r["id"])) \
            if "is_admissible" in r.keys() else None
    conn.commit()

    # добавим колонку допустимости, если её нет
    cols = [c[1] for c in conn.execute("PRAGMA table_info(materials)")]
    if "is_admissible" not in cols:
        conn.execute("ALTER TABLE materials ADD COLUMN is_admissible INTEGER DEFAULT 1")
        for r in conn.execute("SELECT * FROM materials"):
            adm = admissibility(r, r["media_desc"] or "")
            conn.execute("UPDATE materials SET is_admissible=? WHERE id=?", (adm, r["id"]))
        conn.commit()

    # экспорт итогового набора данных (без полного текста — со статами)
    out_csv = DATASET_DIR / "dataset.csv"
    q = conn.execute("""SELECT id, uid, domain, topic, lesson_type, difficulty,
               n_images, n_tables, moderation_passed, moderation_score,
               is_generated, is_admissible, LENGTH(text) tlen, abstract FROM materials""")
    rows = q.fetchall()
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "uid", "domain", "topic", "lesson_type", "difficulty",
                    "n_images", "n_tables", "moderation_passed", "moderation_score",
                    "is_generated", "is_admissible", "text_len", "abstract"])
        for r in rows:
            w.writerow([r["id"], r["uid"], r["domain"], r["topic"], r["lesson_type"],
                        r["difficulty"], r["n_images"], r["n_tables"],
                        r["moderation_passed"], r["moderation_score"],
                        r["is_generated"], r["is_admissible"], r["tlen"],
                        (r["abstract"] or "")[:300]])
    log_event(conn, "A2_dataset", "dataset_built", f"rows={len(rows)}, media_updated={n_media}")
    print(f"Набор данных сформирован: {out_csv} ({len(rows)} записей, "
          f"описания медиа добавлены для {n_media})")
    conn.close()


if __name__ == "__main__":
    main()
