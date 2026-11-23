# -*- coding: utf-8 -*-
"""
Агент B1/B2 (задания 2.1–2.2) — Аналитическая система и дашборд.

Подключается к БД модуля А, рассчитывает ключевые метрики:
 - покрытие тем по дисциплинам относительно среднего;
 - доля сгенерированных материалов (по предметам и в целом);
 - распределение по типам занятий;
 - обеспеченность требований методических рекомендаций по категориям;
 - TOP наиболее/наименее выполняемых требований по дисциплинам;
 - сравнение выполнения требований исходных и сгенерированных материалов.
Строит статический интерактивный HTML-дашборд (обновляется перезапуском) +
PNG-визуализации. Поддерживаются уровни доступа (viewer/analyst/admin).

Запуск: python agent_b1_dashboard.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DASHBOARD_DIR, FIGURES_DIR, REPORTS_DIR, get_conn, log_event)

ACCESS_LEVELS = {  # роль -> доступные разделы
    "viewer": ["kpis", "coverage", "lesson_types"],
    "analyst": ["kpis", "coverage", "generated_share", "lesson_types",
                "requirements", "req_top", "orig_vs_gen"],
    "admin": ["kpis", "coverage", "generated_share", "lesson_types",
              "requirements", "req_top", "orig_vs_gen", "log"],
}


def compute_metrics(conn):
    rows = conn.execute("""SELECT domain, lesson_type, moderation_passed,
        moderation_score, is_generated, moderation_detail, has_prev, has_next,
        difficulty FROM materials""").fetchall()
    n = len(rows)
    by_dom = defaultdict(list)
    for r in rows:
        by_dom[r["domain"]].append(r)

    avg_cov = sum(r["moderation_score"] for r in rows) / n
    coverage = {d: round(sum(r["moderation_score"] for r in rs) / len(rs), 1)
                for d, rs in by_dom.items()}

    gen_share = {d: round(100 * sum(r["is_generated"] for r in rs) / len(rs), 1)
                 for d, rs in by_dom.items()}
    gen_total = round(100 * sum(r["is_generated"] for r in rows) / n, 1)

    lesson_types = dict(Counter(r["lesson_type"] for r in rows))

    # требования по категориям
    req_all = Counter(); req_cnt = Counter()
    req_orig = Counter(); req_orig_n = 0
    req_gen = Counter(); req_gen_n = 0
    req_by_dom = defaultdict(Counter)
    for r in rows:
        d = json.loads(r["moderation_detail"] or "{}")
        for k, v in d.items():
            req_all[k] += 1
            req_cnt[k] += bool(v)
            req_by_dom[r["domain"]][k] += bool(v)
        if r["is_generated"]:
            req_gen_n += 1
            for k, v in d.items():
                req_gen[k] += bool(v)
        else:
            req_orig_n += 1
            for k, v in d.items():
                req_orig[k] += bool(v)
    requirements = {k: round(100 * req_cnt[k] / max(req_all[k], 1), 1)
                    for k in req_all}
    orig_vs_gen = {k: {"original": round(100 * req_orig[k] / max(req_orig_n, 1), 1),
                       "generated": round(100 * req_gen[k] / max(req_gen_n, 1), 1)}
                   for k in req_all}

    req_top = {}
    ndom = {d: len(rs) for d, rs in by_dom.items()}
    for dom, cnt in req_by_dom.items():
        vals = sorted(((round(100 * cnt[k] / ndom[dom], 1), k) for k in cnt))
        req_top[dom] = {"top3": [(k, v) for v, k in vals[-3:]][::-1],
                        "bottom3": [(k, v) for v, k in vals[:3]]}

    return {
        "n_materials": n,
        "avg_moderation": round(avg_cov, 1),
        "passed_share": round(100 * sum(r["moderation_passed"] for r in rows) / n, 1),
        "coverage_vs_avg": {d: {"score": v, "delta_vs_avg": round(v - avg_cov, 1)}
                            for d, v in coverage.items()},
        "generated_share_by_domain": gen_share,
        "generated_share_total": gen_total,
        "lesson_types": lesson_types,
        "requirements_completion": requirements,
        "req_top_bottom_by_domain": req_top,
        "orig_vs_gen": orig_vs_gen,
    }


def make_figures(m):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    cov = m["coverage_vs_avg"]
    names = list(cov)
    vals = [cov[d]["score"] for d in names]
    colors = ["#55a868" if v >= m["avg_moderation"] else "#c44e52" for v in vals]
    axes[0, 0].barh(names, vals, color=colors)
    axes[0, 0].axvline(m["avg_moderation"], color="black", ls="--",
                       label=f"среднее {m['avg_moderation']}")
    axes[0, 0].legend(); axes[0, 0].set_title("Покрытие тем (баллы модерации) vs среднее")

    gs = m["generated_share_by_domain"]
    axes[0, 1].bar(gs.keys(), gs.values(), color="#8172b2")
    axes[0, 1].set_title(f"Доля сгенерированных материалов, % (всего {m['generated_share_total']}%)")
    axes[0, 1].tick_params(axis="x", rotation=45)

    lt = m["lesson_types"]
    axes[1, 0].pie(lt.values(), labels=lt.keys(), autopct="%1.0f%%",
                   colors=["#4c72b0", "#55a868", "#c44e52", "#ccb974"])
    axes[1, 0].set_title("Типы занятий")

    rq = m["requirements_completion"]
    items = sorted(rq.items(), key=lambda kv: kv[1])
    axes[1, 1].barh([k[:38] for k, _ in items], [v for _, v in items], color="#4c72b0")
    axes[1, 1].set_title("Обеспеченность требований метод. рекомендаций, %")
    axes[1, 1].tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dashboard_analytics.png", dpi=110)
    plt.close(fig)


HTML_TMPL = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<title>Дашборд учебных материалов</title>
<meta http-equiv="refresh" content="300">
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f6fa;color:#222}}
header{{background:#1f3b73;color:#fff;padding:14px 24px;display:flex;justify-content:space-between;align-items:center}}
select{{padding:6px;border-radius:6px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;padding:18px}}
.card{{background:#fff;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.12);padding:14px}}
.kpi{{font-size:30px;font-weight:700;color:#1f3b73}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
td,th{{border-bottom:1px solid #e3e6ee;padding:5px 8px;text-align:left}}
.pos{{color:#177a3d}.neg{{color:#b02a2a}} .neg{{color:#b02a2a}}
img{{max-width:100%;border-radius:8px}}
h3{{margin:4px 0 10px}}
</style></head><body>
<header><div><b>MLOKZ Analytics</b> · Дашборд учебных материалов</div>
<div>Роль доступа: <select id="role" onchange="applyRole()">
<option value="viewer">viewer</option><option value="analyst" selected>analyst</option>
<option value="admin">admin</option></select> <span id="ts">{ts}</span></div></header>
<div class="grid">
<div class="card section" data-s="kpis"><h3>Ключевые метрики</h3>
<div class="kpi">{n}</div>материалов в базе<br>
<div class="kpi">{avg}%</div>среднее выполнение метод. требований<br>
<div class="kpi">{passed}%</div>материалов прошли модерацию<br>
<div class="kpi">{gen}%</div>доля сгенерированных материалов</div>
<div class="card section" data-s="coverage"><h3>Покрытие тем относительно среднего</h3>{cov_table}</div>
<div class="card section" data-s="generated_share"><h3>Доля сгенерированных по дисциплинам</h3>{gen_table}</div>
<div class="card section" data-s="lesson_types"><h3>Распределение по типам занятий</h3>{lt_table}</div>
<div class="card section" data-s="requirements"><h3>Обеспеченность требований (категории)</h3>{req_table}</div>
<div class="card section" data-s="req_top"><h3>TOP требований по дисциплинам</h3>{topnotable}</div>
<div class="card section" data-s="orig_vs_gen"><h3>Исходные vs сгенерированные материалы</h3>{ovg_table}</div>
<div class="card section" data-s="log"><h3>Журнал агентов (последние 10)</h3>{agent_log}</div>
<div class="card" style="grid-column:1/-1"><h3>Визуализации</h3>
<img src="../figures/dashboard_analytics.png"><img src="../figures/dataset_structure.png"></div>
</div>
<script>
const ACCESS={access};
function applyRole(){{const r=document.getElementById('role').value;
document.querySelectorAll('.section').forEach(s=>{{s.style.display=ACCESS[r].includes(s.dataset.s)?'':'none';}});}}
applyRole();
</script></body></html>"""


def render_html(m, conn):
    def table(pairs, headers=("Значение", "Показатель")):
        trs = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in pairs)
        return f"<table><tr><th>{headers[0]}</th><th>{headers[1]}</th></tr>{trs}</table>"

    cov_rows = []
    for d, v in sorted(m["coverage_vs_avg"].items(), key=lambda kv: -kv[1]["score"]):
        cls = "pos" if v["delta_vs_avg"] >= 0 else "neg"
        cov_rows.append((d, f"{v['score']} <span class='{cls}'>({v['delta_vs_avg']:+})</span>"))
    topnotable = ""
    for dom, t in m["req_top_bottom_by_domain"].items():
        topnotable += (f"<p><b>{dom}</b><br>↑ " +
                       ", ".join(f"{k} ({v}%)" for k, v in t["top3"]) +
                       "<br>↓ " + ", ".join(f"{k} ({v}%)" for k, v in t["bottom3"]) + "</p>")
    ovg = "<table><tr><th>Требование</th><th>Исходные, %</th><th>Генерированные, %</th></tr>"
    for k, v in m["orig_vs_gen"].items():
        ovg += f"<tr><td>{k[:45]}</td><td>{v['original']}</td><td>{v['generated']}</td></tr>"
    ovg += "</table>"
    logs = conn.execute("SELECT agent,event,detail,created_at FROM agent_log "
                        "ORDER BY id DESC LIMIT 10").fetchall()
    log_tbl = "<table><tr><th>Агент</th><th>Событие</th><th>Детали</th><th>Время</th></tr>" + \
        "".join(f"<tr><td>{l['agent']}</td><td>{l['event']}</td><td>{l['detail'][:60]}</td>"
                f"<td>{l['created_at']}</td></tr>" for l in logs) + "</table>"

    from datetime import datetime
    html = HTML_TMPL.format(
        ts=datetime.now().strftime("%Y-%m-%d %H:%M"),
        n=m["n_materials"], avg=m["avg_moderation"], passed=m["passed_share"],
        gen=m["generated_share_total"],
        cov_table=table(cov_rows, ("Дисциплина", "Балл (Δ к среднему)")),
        gen_table=table(sorted(m["generated_share_by_domain"].items(),
                               key=lambda kv: -kv[1]), ("Дисциплина", "% генерации")),
        lt_table=table(m["lesson_types"].items(), ("Тип занятия", "Кол-во")),
        req_table=table(sorted(m["requirements_completion"].items(),
                               key=lambda kv: kv[1]), ("Требование", "Выполнено, %")),
        topnotable=topnotable, ovg_table=ovg, agent_log=log_tbl,
        access=json.dumps(ACCESS_LEVELS, ensure_ascii=False))
    (DASHBOARD_DIR / "index.html").write_text(html, encoding="utf-8")


def main():
    conn = get_conn()
    m = compute_metrics(conn)
    (REPORTS_DIR / "analytics.json").write_text(
        json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    make_figures(m)
    render_html(m, conn)
    log_event(conn, "B1_dashboard", "rebuilt", f"materials={m['n_materials']}")
    print("Дашборд:", DASHBOARD_DIR / "index.html")
    conn.close()


if __name__ == "__main__":
    main()
