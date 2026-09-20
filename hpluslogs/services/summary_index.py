"""Generate a static index for cached daily / weekly / monthly summaries."""

from __future__ import annotations

import calendar
import datetime as _dt
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from hpluslogs.core.utils import ensure_directory
from hpluslogs.integrations import scp
from hpluslogs.services import summarize


@dataclass(frozen=True)
class SummaryItem:
    kind: str
    label: str
    path: Path
    href: str
    size: int
    mtime: float
    start: str
    end: str
    title: str


def _human_size(size: int) -> str:
    units = ("B", "KB", "MB", "GB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def _month_iter(start: _dt.date, end: _dt.date) -> Iterable[_dt.date]:
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield _dt.date(year, month, 1)
        month += 1
        if month == 13:
            year += 1
            month = 1


def _week_mondays(start: _dt.date, end: _dt.date) -> Iterable[_dt.date]:
    monday = summarize.monday_of(start)
    while monday <= end:
        yield monday
        monday += _dt.timedelta(days=7)


def _first_heading(path: Path) -> str:
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip() or path.stem
    except OSError:
        pass
    return path.stem


def _item(kind: str, label: str, path: Path, start: _dt.date, end: _dt.date) -> SummaryItem:
    stat = path.stat()
    return SummaryItem(
        kind=kind,
        label=label,
        path=path,
        href=f"{kind}-{label}.html",
        size=stat.st_size,
        mtime=stat.st_mtime,
        start=start.isoformat(),
        end=end.isoformat(),
        title=_first_heading(path),
    )


def _scan_generated(data_dir: Path) -> list[SummaryItem]:
    summaries_dir = data_dir / "summaries"
    items: list[SummaryItem] = []

    for path in sorted((summaries_dir / "daily").glob("*.md")):
        try:
            day = _dt.date.fromisoformat(path.stem)
        except ValueError:
            continue
        items.append(_item("daily", path.stem, path, day, day))

    for path in sorted((summaries_dir / "weekly").glob("*.md")):
        monday = summarize._week_label_to_monday(path.stem)
        if monday is None:
            continue
        items.append(_item("weekly", path.stem, path, monday, monday + _dt.timedelta(days=6)))

    for path in sorted((summaries_dir / "monthly").glob("*.md")):
        try:
            year, month = (int(part) for part in path.stem.split("-", 1))
            start = _dt.date(year, month, 1)
            end = _dt.date(year, month, calendar.monthrange(year, month)[1])
        except ValueError:
            continue
        items.append(_item("monthly", path.stem, path, start, end))

    return items


def _raw_days(data_dir: Path) -> list[_dt.date]:
    raw_dir = data_dir / "raw"
    if not raw_dir.exists():
        return []
    days: list[_dt.date] = []
    for path in raw_dir.glob("*.log"):
        try:
            days.append(_dt.date.fromisoformat(path.stem))
        except ValueError:
            continue
    return sorted(set(days))


def _expected_sets(data_dir: Path, generated: list[SummaryItem]) -> dict[str, set[str]]:
    generated_dates = [
        _dt.date.fromisoformat(item.start)
        for item in generated
        if item.kind == "daily"
    ]
    source_days = _raw_days(data_dir) or generated_dates
    if not source_days:
        return {"daily": set(), "weekly": set(), "monthly": set()}

    start, end = min(source_days), max(source_days)
    daily = {day.isoformat() for day in source_days}
    weekly = {summarize.week_label(monday) for monday in _week_mondays(start, end)}
    monthly = {month.strftime("%Y-%m") for month in _month_iter(start, end)}
    return {"daily": daily, "weekly": weekly, "monthly": monthly}


def _json_script(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def _build_html(data_dir: Path, items: list[SummaryItem]) -> str:
    generated = {kind: set() for kind in ("daily", "weekly", "monthly")}
    for item in items:
        generated[item.kind].add(item.label)

    expected = _expected_sets(data_dir, items)
    missing = {
        kind: sorted(expected[kind] - generated[kind])
        for kind in ("daily", "weekly", "monthly")
    }

    daily_items = [item for item in items if item.kind == "daily"]
    latest = max((item.mtime for item in items), default=0)
    largest = max((item.size for item in items), default=0)
    total_size = sum(item.size for item in items)
    generated_at = _dt.datetime.now().astimezone().isoformat(timespec="seconds")

    payload = {
        "items": [
            {
                "kind": item.kind,
                "label": item.label,
                "href": item.href,
                "size": item.size,
                "sizeText": _human_size(item.size),
                "mtime": item.mtime,
                "modified": _dt.datetime.fromtimestamp(item.mtime).astimezone().isoformat(timespec="seconds"),
                "start": item.start,
                "end": item.end,
                "title": item.title,
            }
            for item in sorted(items, key=lambda x: (x.kind, x.label))
        ],
        "missing": missing,
        "expected": {kind: sorted(values) for kind, values in expected.items()},
    }

    counts_html = "\n".join(
        f"""
        <article class="stat">
          <span>{kind.title()}</span>
          <strong>{len(generated[kind])}</strong>
          <small>{len(missing[kind])} missing</small>
        </article>"""
        for kind in ("daily", "weekly", "monthly")
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>hplusroadmap Chat Summary Index</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f7f5ef;
      --panel: #fffdfa;
      --ink: #161616;
      --muted: #64615a;
      --line: #ded8cc;
      --accent: #0f766e;
      --accent-2: #8b5a2b;
      --missing: #b42318;
      --soft: #e7f4f1;
      --visited: #6d3f8f;
      --shadow: 0 18px 45px rgba(28, 25, 21, 0.10);
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #181713;
        --panel: #22201b;
        --ink: #f4efe6;
        --muted: #b9b0a2;
        --line: #3c372f;
        --accent: #5eead4;
        --accent-2: #f1b36a;
        --missing: #ff8a80;
        --soft: #14322e;
        --visited: #d8b4fe;
        --shadow: 0 18px 45px rgba(0, 0, 0, 0.28);
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 15px/1.5 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    a {{ color: #065f8f; text-decoration-thickness: 0.09em; text-underline-offset: 0.18em; }}
    a:visited {{ color: var(--visited); }}
    a:hover {{ color: var(--accent); }}
    header {{
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    .wrap {{ width: min(1180px, calc(100% - 32px)); margin: 0 auto; }}
    .hero {{ padding: 32px 0 24px; }}
    h1 {{ margin: 0; font-size: clamp(28px, 4vw, 54px); line-height: 1.02; letter-spacing: 0; }}
    .subtitle {{ margin: 12px 0 0; max-width: 820px; color: var(--muted); font-size: 17px; }}
    .toolbar {{
      position: sticky;
      top: 0;
      z-index: 5;
      background: color-mix(in srgb, var(--panel) 94%, transparent);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(10px);
    }}
    .toolbar-inner {{
      display: grid;
      grid-template-columns: minmax(220px, 1fr) auto auto auto;
      gap: 10px;
      align-items: center;
      padding: 12px 0;
    }}
    input, select, button {{
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--ink);
      font: inherit;
    }}
    input {{ width: 100%; padding: 0 12px; }}
    select, button {{ padding: 0 10px; }}
    button {{ cursor: pointer; }}
    button.active {{ border-color: var(--accent); background: var(--soft); color: var(--ink); }}
    .segmented {{
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 4px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: color-mix(in srgb, var(--panel) 86%, var(--bg));
    }}
    .segmented button {{
      min-height: 30px;
      border: 0;
      background: transparent;
      white-space: nowrap;
    }}
    .segmented button.active {{
      background: var(--soft);
      box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--accent) 46%, transparent);
    }}
    .toolbar-status {{
      grid-column: 1 / -1;
      display: flex;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
      color: var(--muted);
      font-size: 13px;
    }}
    #clearFilters {{ color: var(--accent); }}
    main {{ padding: 24px 0 48px; }}
    .stats {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; margin-bottom: 22px; }}
    .stat {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      box-shadow: var(--shadow);
    }}
    .stat span, .stat small {{ display: block; color: var(--muted); }}
    .stat strong {{ display: block; margin: 4px 0 2px; font-size: 28px; line-height: 1; }}
    .tabs {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 18px; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: var(--shadow);
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); cursor: pointer; }}
    tr:last-child td {{ border-bottom: 0; }}
    .kind {{ display: inline-block; min-width: 68px; color: var(--accent-2); font-weight: 700; }}
    .muted {{ color: var(--muted); }}
    .missing-link {{ color: var(--missing); font-weight: 700; }}
    .calendar {{ display: grid; gap: 18px; }}
    .year {{ padding: 16px; }}
    .year h2, .missing h2 {{ margin: 0 0 12px; font-size: 22px; }}
    .months {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 14px; }}
    .month {{ border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: color-mix(in srgb, var(--panel) 92%, var(--soft)); }}
    .month h3 {{ margin: 0 0 8px; font-size: 15px; }}
    .dow, .days {{ display: grid; grid-template-columns: repeat(7, 1fr); gap: 4px; }}
    .dow span {{ color: var(--muted); font-size: 11px; text-align: center; }}
    .day {{
      display: grid;
      place-items: center;
      aspect-ratio: 1;
      min-width: 0;
      border-radius: 6px;
      border: 1px solid var(--line);
      font-size: 12px;
      text-decoration: none;
    }}
    .day.hit {{ background: var(--soft); border-color: color-mix(in srgb, var(--accent) 42%, var(--line)); font-weight: 700; }}
    .day.miss {{ color: var(--muted); opacity: 0.55; }}
    .day.blank {{ border-color: transparent; background: transparent; }}
    .missing-list {{ display: flex; flex-wrap: wrap; gap: 8px; padding: 16px; }}
    .pill {{ border: 1px solid var(--line); border-radius: 999px; padding: 5px 10px; color: var(--missing); background: color-mix(in srgb, var(--panel) 90%, var(--missing)); }}
    .hidden {{ display: none !important; }}
    @media (max-width: 840px) {{
      .toolbar-inner {{ grid-template-columns: 1fr; }}
      .toolbar-inner input {{ grid-column: 1 / -1; }}
      .segmented {{ width: 100%; overflow-x: auto; justify-content: flex-start; }}
      .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      th:nth-child(4), td:nth-child(4) {{ display: none; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap hero">
      <h1>hplusroadmap Chat Summary Index</h1>
      <p class="subtitle">Daily, weekly, and monthly summaries generated from IRC logs. Use the controls to find summaries, sort by freshness or size, and switch to the calendar or missing-only view.</p>
    </div>
  </header>
  <div class="toolbar">
    <div class="wrap toolbar-inner">
      <input id="search" type="search" placeholder="Search labels, titles, dates...">
      <div class="segmented" id="kindFilters" aria-label="Summary type">
        <button class="active" type="button" data-kind="all">All</button>
        <button type="button" data-kind="daily">Daily</button>
        <button type="button" data-kind="weekly">Weekly</button>
        <button type="button" data-kind="monthly">Monthly</button>
      </div>
      <select id="sort">
        <option value="date-desc">Newest period</option>
        <option value="date-asc">Oldest period</option>
        <option value="mtime-desc">Latest modified</option>
        <option value="mtime-asc">Oldest modified</option>
        <option value="size-desc">Largest files</option>
        <option value="size-asc">Smallest files</option>
        <option value="label-asc">Label A-Z</option>
        <option value="label-desc">Label Z-A</option>
        <option value="kind-asc">Type A-Z</option>
        <option value="kind-desc">Type Z-A</option>
        <option value="start-desc">Coverage newest</option>
        <option value="start-asc">Coverage oldest</option>
      </select>
      <div class="segmented" id="viewFilters" aria-label="View">
        <button id="tableTab" class="active" type="button" data-view="list">List</button>
        <button id="calendarTab" type="button" data-view="calendar">Calendar</button>
        <button id="missingTab" type="button" data-view="missing">Missing</button>
      </div>
      <div class="toolbar-status">
        <span id="resultStatus">Loading summaries...</span>
        <button id="clearFilters" type="button">Clear filters</button>
      </div>
    </div>
  </div>
  <main class="wrap">
    <section class="stats" aria-label="Summary statistics">
      {counts_html}
      <article class="stat"><span>Total files</span><strong>{len(items)}</strong><small>{_human_size(total_size)} total</small></article>
      <article class="stat"><span>Largest</span><strong>{_human_size(largest)}</strong><small>single markdown file</small></article>
      <article class="stat"><span>Generated</span><strong>{html.escape(generated_at[:10])}</strong><small>{html.escape(generated_at[11:19])}</small></article>
    </section>
    <section id="listView" class="panel">
      <table>
        <thead><tr>
          <th data-sort="kind">Type</th>
          <th data-sort="label">Summary</th>
          <th data-sort="size">Size</th>
          <th data-sort="mtime">Modified</th>
          <th data-sort="start">Coverage</th>
        </tr></thead>
        <tbody id="rows"></tbody>
      </table>
    </section>
    <section id="calendarView" class="calendar hidden" aria-label="Daily summary calendar"></section>
    <section id="missingView" class="panel missing hidden" aria-label="Missing summaries"></section>
  </main>
  <script id="summary-data" type="application/json">{_json_script(payload)}</script>
  <script>
    const data = JSON.parse(document.getElementById('summary-data').textContent);
    const rows = document.getElementById('rows');
    const search = document.getElementById('search');
    const sort = document.getElementById('sort');
    const resultStatus = document.getElementById('resultStatus');
    const clearFilters = document.getElementById('clearFilters');
    let kindMode = 'all';
    let viewMode = 'list';
    const views = {{
      list: document.getElementById('listView'),
      calendar: document.getElementById('calendarView'),
      missing: document.getElementById('missingView'),
    }};
    const buttons = {{
      list: document.getElementById('tableTab'),
      calendar: document.getElementById('calendarTab'),
      missing: document.getElementById('missingTab'),
    }};

    function fmtDate(value) {{
      return new Date(value).toLocaleString([], {{ dateStyle: 'medium', timeStyle: 'short' }});
    }}

    function filteredItems() {{
      const q = search.value.trim().toLowerCase();
      let items = data.items.filter(item => kindMode === 'all' || item.kind === kindMode);
      if (q) {{
        items = items.filter(item => `${{item.kind}} ${{item.label}} ${{item.title}} ${{item.start}} ${{item.end}}`.toLowerCase().includes(q));
      }}
      const mode = sort.value;
      const dir = mode.endsWith('asc') ? 1 : -1;
      const key = mode.replace(/-(asc|desc)$/, '');
      items.sort((a, b) => {{
        if (key === 'date') return dir * a.start.localeCompare(b.start);
        if (key === 'mtime' || key === 'size') return dir * (a[key] - b[key]);
        return dir * String(a[key]).localeCompare(String(b[key]));
      }});
      return items;
    }}

    function renderRows() {{
      const items = filteredItems();
      resultStatus.textContent = statusText(items.length);
      rows.innerHTML = items.map(item => `
        <tr>
          <td><span class="kind">${{item.kind}}</span></td>
          <td><a href="${{item.href}}">${{item.label}}</a><div class="muted">${{escapeHtml(item.title)}}</div></td>
          <td data-size="${{item.size}}">${{item.sizeText}}</td>
          <td>${{fmtDate(item.modified)}}</td>
          <td>${{item.start === item.end ? item.start : item.start + ' to ' + item.end}}</td>
        </tr>`).join('');
    }}

    function statusText(count) {{
      const typeText = kindMode === 'all' ? 'all summary types' : kindMode + ' summaries';
      const searchText = search.value.trim() ? ` matching "${{search.value.trim()}}"` : '';
      return `${{count.toLocaleString()}} generated ${{typeText}}${{searchText}}`;
    }}

    function escapeHtml(text) {{
      return String(text).replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
    }}

    function setView(name) {{
      viewMode = name;
      Object.entries(views).forEach(([key, el]) => el.classList.toggle('hidden', key !== name));
      Object.entries(buttons).forEach(([key, el]) => el.classList.toggle('active', key === name));
      renderAll();
    }}

    function setKind(name) {{
      kindMode = name;
      document.querySelectorAll('#kindFilters button').forEach(btn => {{
        btn.classList.toggle('active', btn.dataset.kind === name);
      }});
      renderAll();
    }}

    function renderAll() {{
      renderRows();
      if (viewMode === 'calendar') renderCalendar();
      if (viewMode === 'missing') renderMissing();
    }}

    function renderCalendar() {{
      if (kindMode !== 'all' && kindMode !== 'daily') {{
        views.calendar.innerHTML = `<section class="panel year"><h2>Calendar view is for daily summaries</h2><p class="muted">Use List or Missing to inspect ${{kindMode}} summaries, or choose All/Daily to return to the daily calendar.</p></section>`;
        return;
      }}
      const daily = new Map(data.items.filter(item => item.kind === 'daily').map(item => [item.label, item]));
      const expected = data.expected.daily.length ? data.expected.daily : [...daily.keys()];
      if (!expected.length) {{
        views.calendar.innerHTML = '<section class="panel year"><h2>No daily summaries found</h2></section>';
        return;
      }}
      const groups = new Map();
      for (const label of expected) {{
        const d = new Date(label + 'T12:00:00');
        const ym = label.slice(0, 7);
        if (!groups.has(ym)) groups.set(ym, []);
        groups.get(ym).push({{ label, day: d.getDate(), dow: d.getDay(), item: daily.get(label) }});
      }}
      const years = new Map();
      for (const [ym, days] of groups) {{
        const year = ym.slice(0, 4);
        if (!years.has(year)) years.set(year, []);
        years.get(year).push([ym, days]);
      }}
      views.calendar.innerHTML = [...years].sort().reverse().map(([year, months]) => `
        <section class="panel year">
          <h2>${{year}}</h2>
          <div class="months">
            ${{months.sort().map(([ym, days]) => monthHtml(ym, days)).join('')}}
          </div>
        </section>`).join('');
    }}

    function monthHtml(ym, days) {{
      const first = days[0];
      const blanks = Array.from({{ length: first.dow }}, () => '<span class="day blank"></span>').join('');
      const cells = days.map(d => d.item
        ? `<a class="day hit" href="${{d.item.href}}" title="${{escapeHtml(d.item.title)}}">${{d.day}}</a>`
        : `<span class="day miss" title="Missing ${{d.label}}">${{d.day}}</span>`).join('');
      return `<article class="month">
        <h3>${{new Date(ym + '-01T12:00:00').toLocaleString([], {{ month: 'long', year: 'numeric' }})}}</h3>
        <div class="dow"><span>S</span><span>M</span><span>T</span><span>W</span><span>T</span><span>F</span><span>S</span></div>
        <div class="days">${{blanks}}${{cells}}</div>
      </article>`;
    }}

    function renderMissing() {{
      const types = kindMode === 'all' ? ['daily', 'weekly', 'monthly'] : [kindMode];
      views.missing.innerHTML = types.map(type => {{
        const values = data.missing[type] || [];
        const body = values.length
          ? `<div class="missing-list">${{values.map(v => `<span class="pill">${{v}}</span>`).join('')}}</div>`
          : '<div class="missing-list"><span class="muted">No missing summaries in the expected range.</span></div>';
        return `<section><h2>${{type[0].toUpperCase() + type.slice(1)}} missing</h2>${{body}}</section>`;
      }}).join('');
    }}

    document.querySelectorAll('th[data-sort]').forEach(th => {{
      th.addEventListener('click', () => {{
        const key = th.dataset.sort;
        const current = sort.value;
        sort.value = current === `${{key}}-asc` ? `${{key}}-desc` : `${{key}}-asc`;
        renderAll();
      }});
    }});
    document.querySelectorAll('#kindFilters button').forEach(btn => {{
      btn.addEventListener('click', () => setKind(btn.dataset.kind));
    }});
    search.addEventListener('input', renderAll);
    sort.addEventListener('change', renderAll);
    buttons.list.addEventListener('click', () => setView('list'));
    buttons.calendar.addEventListener('click', () => setView('calendar'));
    buttons.missing.addEventListener('click', () => setView('missing'));
    clearFilters.addEventListener('click', () => {{
      search.value = '';
      sort.value = 'date-desc';
      setKind('all');
      setView('list');
    }});
    renderAll();
    renderCalendar();
    renderMissing();
  </script>
</body>
</html>
"""


def generate(
    data_dir: Path,
    upload: bool = True,
    remote_user: str = "bryan",
    remote_host: str = "gnusha.org",
    remote_path: str = summarize.DEFAULT_REMOTE_PATH,
) -> Path:
    """Generate ``data_dir/summaries/index.html`` and optionally upload it."""
    items = _scan_generated(data_dir)
    summaries_dir = data_dir / "summaries"
    ensure_directory(summaries_dir)
    output_path = summaries_dir / "index.html"
    output_path.write_text(_build_html(data_dir, items), encoding="utf-8")

    if upload:
        scp.ensure_remote_directory(remote_user, remote_host, remote_path)
        ok = scp.upload_file(output_path, remote_user, remote_host, remote_path, "index.html")
        if not ok:
            raise RuntimeError(f"failed to upload {output_path} to {remote_user}@{remote_host}:{remote_path}")

    return output_path
