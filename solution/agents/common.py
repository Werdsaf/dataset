# -*- coding: utf-8 -*-
"""
Общий модуль мультиагентной системы анализа учебных материалов.
Содержит: конфигурацию путей, работу с БД (SQLite), утилиты TF-IDF,
метрику Жаккара, модерацию материалов по методическим рекомендациям.
Работает полностью офлайн, без внешних сетевых зависимостей.
"""
import json
import math
import os
import re
import sqlite3
from pathlib import Path

# ---------------------------------------------------------------- пути
HERE = Path(__file__).resolve().parent
SOLUTION_DIR = HERE.parent
OUTPUT_DIR = SOLUTION_DIR / "output"
DB_PATH = SOLUTION_DIR / "db" / "materials.db"
DATASET_DIR = SOLUTION_DIR / "dataset"          # расширенный датасет (csv/json)
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"
DASHBOARD_DIR = OUTPUT_DIR / "dashboard"

SOURCE_DATASET = SOLUTION_DIR.parent / "summarization-dataset-main" / "dataset"

for d in (OUTPUT_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, DASHBOARD_DIR,
          DATASET_DIR, DB_PATH.parent):
    d.mkdir(parents=True, exist_ok=True)

# предметные области исходного корпуса
DOMAINS = {
    "chemistry": "Химия",
    "economics": "Экономика",
    "history": "История",
    "it": "Информатика",
    "journalism": "Журналистика",
    "law": "Право",
    "linguistics": "Лингвистика",
    "medicine": "Медицина",
}

# ------------------------------------------------------- текстовые утилиты
STOPWORDS_RU = set("""
и в во не что он на я с со как а все она то его но да ты к у ее вы за было
бы только еще м н было есть о нем так вам ну ли л мне от меня о ней них еще
можно нужно может быть очень здесь этот эти этот когда где если чтобы нас
вас нее него их нее тем том чем как же же уж
""".split())

WORD_RE = re.compile(r"[а-яёА-ЯЁa-zA-Z0-9\-]+")


def tokenize(text: str):
    """Простая русско-английская токенизация без внешних библиотек."""
    toks = [w.lower() for w in WORD_RE.findall(text)]
    return [t for t in toks if len(t) > 2 and t not in STOPWORDS_RU]


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# --------------------------------------------------------------- схема БД
SCHEMA = """
CREATE TABLE IF NOT EXISTS materials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uid TEXT UNIQUE NOT NULL,            -- идентификатор материала
    source_url TEXT,                     -- ссылка/путь источника
    domain TEXT NOT NULL,                -- предмет (дисциплина)
    topic TEXT,                          -- тема (название)
    text TEXT,                           -- текстовый материал
    abstract TEXT,                       -- краткая аннотация
    lesson_type TEXT,                    -- тип занятия
    difficulty REAL,                     -- сложность 1..5
    est_hours REAL,                      -- оценочное время освоения (ч)
    n_images INTEGER DEFAULT 0,          -- число иллюстраций
    n_tables INTEGER DEFAULT 0,          -- число таблиц
    media_desc TEXT,                     -- описания нетекстовых составляющих (JSON)
    moderation_passed INTEGER,           -- итог модерации 0/1
    moderation_score REAL,               -- % выполнения требований
    moderation_detail TEXT,              -- детали по требованиям (JSON)
    is_generated INTEGER DEFAULT 0,      -- материал сгенерирован агентом
    has_prev INTEGER DEFAULT 0,          -- наличие предыдущего по теме материала
    has_next INTEGER DEFAULT 0,          -- наличие последующего по теме материала
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS labels (
    material_id INTEGER PRIMARY KEY,
    parallel_cluster INTEGER,            -- кластер для параллельного изучения
    sequence_cluster INTEGER,            -- кластер для последовательного изучения
    difficulty_cluster INTEGER,          -- кластер сложности
    FOREIGN KEY (material_id) REFERENCES materials(id)
);
CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT, task TEXT, algo TEXT,
    metrics TEXT,                        -- JSON
    n_train INTEGER, drift REAL,
    path TEXT, created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS agent_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent TEXT, event TEXT, detail TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
"""


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def log_event(conn, agent, event, detail=""):
    conn.execute("INSERT INTO agent_log(agent,event,detail) VALUES(?,?,?)",
                 (agent, event, detail))
    conn.commit()


# -------------------------------------------------- методические требования
# Категории требований к учебным материалам (методические рекомендации).
METHOD_REQUIREMENTS = [
    ("Наличие аннотации", lambda m: bool(m["abstract"]) and len(m["abstract"]) >= 80),
    ("Наличие темы/названия", lambda m: bool(m["topic"]) and len(m["topic"]) >= 10),
    ("Достаточный объем текста (>2000 зн.)", lambda m: len(m["text"]) > 2000),
    ("Структурность (заголовки/абзацы)", lambda m: m["text"].count("\n\n") >= 3),
    ("Наличие иллюстративного материала", lambda m: (m["n_images"] + m["n_tables"]) > 0),
    ("Описание медиакомпонентов", lambda m: bool(m.get("media_desc"))),
    ("Тематическая связность (термины повторяются)", lambda m: m.get("_cohesion", 0) > 0.05),
    ("Отсутствие запрещенного контента", lambda m: not m.get("_banned", False)),
    ("Наличие ключевых слов (>=5 уникальных)", lambda m: m.get("_n_keywords", 0) >= 5),
    ("Завершенность (есть выводы)", lambda m: bool(re.search(
        r"(вывод|итог|таким образом|в заключени|результат)", m["text"][-1500:], re.I))),
]

BANNED_PATTERNS = re.compile(r"(спам|казино|порно|наркотики)", re.I)


def moderate(material: dict) -> dict:
    """Модерация материала по методическим рекомендациям.

    Возвращает dict(passed, score, detail{требование: bool}).
    Проходным считается материал с выполнением >= 70% требований.
    """
    tokens = tokenize(material["text"])
    uniq = set(tokens)
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    top = sorted(freq.items(), key=lambda x: -x[1])[:10]
    cohesion = sum(c for _, c in top) / max(len(tokens), 1)

    material["_cohesion"] = cohesion
    material["_n_keywords"] = len([k for k, c in top if c >= 3])
    material["_banned"] = bool(BANNED_PATTERNS.search(material["text"][:4000]))

    detail = {}
    for name, check in METHOD_REQUIREMENTS:
        try:
            detail[name] = bool(check(material))
        except Exception:
            detail[name] = False
    score = 100.0 * sum(detail.values()) / len(detail)
    return {"passed": int(score >= 70), "score": round(score, 1), "detail": detail}


# --------------------------------------------- признаки схожести (TF-IDF)
def build_tfidf(docs, max_features=400):
    """Лёгкий TF-IDF без sklearn (для агентов сбора/разметки)."""
    docs_tokens = [tokenize(d) for d in docs]
    df = {}
    for toks in docs_tokens:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    vocab = [w for w, c in sorted(df.items(), key=lambda x: -x[1])[:max_features]]
    vidx = {w: i for i, w in enumerate(vocab)}
    n = len(docs)
    mat = []
    for toks in docs_tokens:
        tf = {}
        for t in toks:
            if t in vidx:
                tf[t] = tf.get(t, 0) + 1
        vec = [0.0] * len(vocab)
        for t, c in tf.items():
            vec[vidx[t]] = (1 + math.log(c)) * math.log(n / (1 + df[t]))
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        mat.append([v / norm for v in vec])
    return mat, vocab


def cosine(u, v):
    return sum(a * b for a, b in zip(u, v))


if __name__ == "__main__":
    init_db()
    print(f"БД инициализирована: {DB_PATH}")
