"""Summarization service — daily / weekly / monthly digests of the IRC logs.

A daily summary reads one day's raw IRC log, follows any links posted that day
to gather details about manuscripts/papers/resources, and asks an LLM (DeepSeek
via OpenRouter by default) to produce a markdown digest that specifically calls
out posted manuscripts and interesting technical discussions.

Weekly summaries consolidate the seven daily summaries of an ISO week; monthly
summaries consolidate the weekly summaries overlapping a calendar month. Each
summary is saved canonically under ``data_dir/summaries/`` (so aggregations can
reuse them) and also published as markdown + HTML and uploaded to the remote
``chatsummaries/`` directory, mirroring the other CLI commands.
"""

from __future__ import annotations

import calendar
import datetime as _dt
import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import click

from hpluslogs.core.prompts import (
    build_daily_summary_prompt,
    build_monthly_summary_prompt,
    build_weekly_summary_prompt,
)
from hpluslogs.core.utils import daterange, ensure_directory, read_human_text, token_count
from hpluslogs.integrations import gnusha, linkfetch, openrouter
from hpluslogs.services import publishing

LOG_URL_TEMPLATE = "https://gnusha.org/logs/{date}.log"
DEFAULT_CONCURRENCY = 10

# Serializes stdout so buffered per-day log blocks are printed atomically when
# daily summaries are generated concurrently.
_PRINT_LOCK = threading.Lock()

# Control characters (except tab/newline) get stripped before printing.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _sanitize(msg: str) -> str:
    """Make a string safe to print to stdout regardless of terminal encoding.

    Strips control characters and, if the terminal encoding cannot represent
    some characters, replaces them rather than letting ``click.echo`` raise a
    ``UnicodeEncodeError`` (which would break the run).
    """
    if not isinstance(msg, str):
        msg = str(msg)
    msg = _CONTROL_CHARS_RE.sub("", msg)
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        msg.encode(enc)
    except (UnicodeEncodeError, LookupError):
        msg = msg.encode(enc, errors="replace").decode(enc, errors="replace")
    return msg


def _fmt(msg: str) -> str:
    """Sanitize and prepend a wall-clock timestamp to a single-line log message.

    Multi-line blocks (e.g. an echoed summary) and blank spacers are left
    unstamped so content stays clean.
    """
    s = _sanitize(msg)
    if s and "\n" not in s:
        return time.strftime("[%H:%M:%S] ") + s
    return s


def _echo(msg: str = "") -> None:
    """Timestamping, sanitizing wrapper around click.echo for all summarizer stdout."""
    click.echo(_fmt(msg))


def _echo_locked(msg: str = "") -> None:
    """Thread-safe single-line echo for live progress from parallel workers."""
    with _PRINT_LOCK:
        _echo(msg)


def _flush_lines(lines: List[str]) -> None:
    """Print a buffered block of already-formatted log lines atomically."""
    if not lines:
        return
    with _PRINT_LOCK:
        for line in lines:
            click.echo(line)


def _run_day_jobs(
    jobs: List[Tuple[_dt.date, Callable]],
    concurrency: int,
    *,
    buffer_output: bool = True,
) -> dict:
    """Run jobs, up to ``concurrency`` at a time. Returns {day: summary}.

    Each job is ``(day, run_fn)`` where ``run_fn(emit)`` generates the summary and
    sends log lines to the ``emit`` callback. By default, parallel jobs buffer
    their lines and flush them as an atomic block on completion, so interleaved
    daily output stays readable. Set ``buffer_output=False`` for long-running
    higher-level jobs where live progress is more useful than grouped logs.
    """
    results: dict = {}
    if not jobs:
        return results
    workers = max(1, min(concurrency, len(jobs)))

    if workers == 1:
        for day, run_fn in jobs:
            summary = run_fn(_echo)
            if summary is not None:
                results[day] = summary
        return results

    if not buffer_output:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fut_map = {pool.submit(run_fn, _echo_locked): day for day, run_fn in jobs}
            for fut in as_completed(fut_map):
                day = fut_map[fut]
                summary = fut.result()
                if summary is not None:
                    results[day] = summary
        return results

    def worker(run_fn: Callable):
        lines: List[str] = []
        # Timestamp each line when it is produced (not when the block is flushed).
        summary = run_fn(lambda m: lines.append(_fmt(m)))
        return summary, lines

    with ThreadPoolExecutor(max_workers=workers) as pool:
        fut_map = {pool.submit(worker, run_fn): day for day, run_fn in jobs}
        for fut in as_completed(fut_map):
            day = fut_map[fut]
            summary, lines = fut.result()
            _flush_lines(lines)
            if summary is not None:
                results[day] = summary
    return results

DEFAULT_MODEL = "openrouter/deepseek/deepseek-v4-flash"
DEFAULT_REMOTE_PATH = "~/public_html/irc/chatgpt/hplusroadmap/chatsummaries/"


###############################################################################
# Output cleanup
###############################################################################

# Matches an LLM response that wraps the whole markdown document in a single
# ```markdown ... ``` fence, optionally preceded by one conversational preamble
# line ("Here's the summary:"). The closing fence must be at the very end so we
# don't mistakenly unwrap an embedded code snippet.
_WRAPPED_RE = re.compile(r"^(?:[^\n]*\n\n?)?```(?:markdown|md)?\s*\n(.*)\n```\s*$", re.DOTALL)


def _clean_summary(text: str) -> str:
    """Strip stray code-fence wrappers / preambles some models add around output."""
    text = text.strip()
    match = _WRAPPED_RE.match(text)
    if match and match.group(1).lstrip().startswith("#"):
        return match.group(1).strip()
    # Fallback: a bare leading fence with no trailing fence at EOF.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


# Resume threshold. A real summary is always thousands of characters; anything
# shorter than this is treated as empty/truncated (a transient API hiccup) — such
# a file is NEVER saved over a good one, and on a later run it is regenerated
# rather than reused. Existing files at/above this length are considered "done"
# and skipped on resume (unless a --force flag is given).
MIN_SUMMARY_CHARS = 50
LLM_RETRIES = 3
LLM_RETRY_DELAY = 3.0  # seconds between retries
LLM_TIMEOUT = 600.0    # seconds before a single LLM call is aborted as hung
LLM_HEARTBEAT = 20.0   # seconds between "still waiting" progress notices


def _call_with_heartbeat(prompt: str, model: str, emit: Callable) -> str:
    """Call the LLM with a timeout, emitting a heartbeat while we wait.

    The heartbeat makes it obvious we are blocked on the API (not hung), and the
    timeout guarantees the call cannot block forever.
    """
    stop = threading.Event()

    def _beat() -> None:
        waited = 0.0
        while not stop.wait(LLM_HEARTBEAT):
            waited += LLM_HEARTBEAT
            emit(f"    … still waiting on {model} ({int(waited)}s elapsed)…")

    beater = threading.Thread(target=_beat, daemon=True)
    beater.start()
    try:
        return openrouter.complete(prompt, model, timeout=LLM_TIMEOUT) or ""
    finally:
        stop.set()
        beater.join(timeout=1.0)


def _complete_checked(prompt: str, model: str, emit: Callable) -> str:
    """Call the LLM and return a cleaned, non-empty summary, retrying on empties.

    Returns "" only if every attempt produced an empty/too-short response, so the
    caller can refuse to overwrite an existing good file.
    """
    result = ""
    for attempt in range(1, LLM_RETRIES + 1):
        raw = ""
        failed = False
        t0 = time.monotonic()
        try:
            raw = _call_with_heartbeat(prompt, model, emit)
        except Exception as exc:  # noqa: BLE001 - transient API/network/timeout errors
            failed = True
            emit(f"  ⚠ LLM call failed after {time.monotonic() - t0:.1f}s "
                 f"(attempt {attempt}/{LLM_RETRIES}): {exc}")
        elapsed = time.monotonic() - t0

        cleaned = _clean_summary(raw)
        # If cleaning stripped a substantial response to nothing, keep the raw text.
        if not cleaned.strip() and raw.strip():
            cleaned = raw.strip()
        if len(cleaned.strip()) >= MIN_SUMMARY_CHARS:
            emit(f"  Received {len(cleaned.strip())} chars from {model} in {elapsed:.1f}s.")
            return cleaned

        result = cleaned
        if not failed:
            emit(f"  ⚠ Model returned an empty/too-short response "
                 f"({len(cleaned.strip())} chars) after {elapsed:.1f}s.")
        if attempt < LLM_RETRIES:
            emit(f"  Retrying in {LLM_RETRY_DELAY:.0f}s (attempt {attempt + 1}/{LLM_RETRIES})…")
            time.sleep(LLM_RETRY_DELAY)
        else:
            emit(f"  ✗ Giving up after {LLM_RETRIES} attempts (leaving any existing file intact).")
    return result


def _usable(text: Optional[str]) -> bool:
    """True if ``text`` looks like a real summary (not empty/truncated)."""
    return bool(text) and len(text.strip()) >= MIN_SUMMARY_CHARS


def _existing_if_usable(path: Path) -> Optional[str]:
    """Return the file's content only if it exists and is a usable summary."""
    if path.exists():
        text = read_human_text(path)
        if _usable(text):
            return text
    return None


def _is_stale(parent: Path, children: List[Path]) -> bool:
    """True if ``parent`` is missing or older than any existing ``children``.

    Lets resume detect across separate runs that a consolidated summary is out of
    date because an underlying summary was regenerated after it was last built
    (e.g. a weekly was fixed later, so the monthly must be rebuilt).
    """
    if not parent.exists():
        return True
    parent_mtime = parent.stat().st_mtime
    for child in children:
        if child.exists() and child.stat().st_mtime > parent_mtime:
            return True
    return False


###############################################################################
# Paths & labels
###############################################################################

def _summaries_dir(data_dir: Path) -> Path:
    return data_dir / "summaries"


def _daily_path(data_dir: Path, day: _dt.date) -> Path:
    return _summaries_dir(data_dir) / "daily" / f"{day.isoformat()}.md"


def _weekly_path(data_dir: Path, monday: _dt.date) -> Path:
    return _summaries_dir(data_dir) / "weekly" / f"{week_label(monday)}.md"


def _monthly_path(data_dir: Path, year: int, month: int) -> Path:
    return _summaries_dir(data_dir) / "monthly" / f"{year:04d}-{month:02d}.md"


def monday_of(day: _dt.date) -> _dt.date:
    """Return the Monday of the ISO week containing ``day``."""
    return day - _dt.timedelta(days=day.weekday())


def week_label(day: _dt.date) -> str:
    """ISO week label like ``2026-W17`` for the week containing ``day``."""
    iso_year, iso_week, _ = day.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"


def week_days(monday: _dt.date) -> List[_dt.date]:
    """The seven dates Monday..Sunday of the week starting at ``monday``."""
    return [monday + _dt.timedelta(days=i) for i in range(7)]


def month_weeks(year: int, month: int) -> List[_dt.date]:
    """Mondays of all ISO weeks that overlap the given calendar month."""
    last_day = calendar.monthrange(year, month)[1]
    mondays: List[_dt.date] = []
    seen = set()
    for d in range(1, last_day + 1):
        m = monday_of(_dt.date(year, month, d))
        if m not in seen:
            seen.add(m)
            mondays.append(m)
    return mondays


def _daily_summary_days(data_dir: Path) -> List[_dt.date]:
    """Return dates with usable cached daily summaries."""
    daily_dir = _summaries_dir(data_dir) / "daily"
    if not daily_dir.exists():
        return []
    days: List[_dt.date] = []
    for path in daily_dir.glob("*.md"):
        try:
            day = _dt.date.fromisoformat(path.stem)
        except ValueError:
            continue
        if _existing_if_usable(path) is not None:
            days.append(day)
    return sorted(days)


def _week_label_to_monday(label: str) -> Optional[_dt.date]:
    """Parse an ISO week label like ``2026-W17`` into its Monday date."""
    match = re.fullmatch(r"(\d{4})-W(\d{2})", label)
    if not match:
        return None
    try:
        return _dt.date.fromisocalendar(int(match.group(1)), int(match.group(2)), 1)
    except ValueError:
        return None


def _weekly_summary_mondays(data_dir: Path) -> List[_dt.date]:
    """Return ISO-week Mondays with usable cached weekly summaries."""
    weekly_dir = _summaries_dir(data_dir) / "weekly"
    if not weekly_dir.exists():
        return []
    mondays: List[_dt.date] = []
    for path in weekly_dir.glob("*.md"):
        monday = _week_label_to_monday(path.stem)
        if monday is not None and _existing_if_usable(path) is not None:
            mondays.append(monday)
    return sorted(set(mondays))


def _months_overlapped_by_weeks(mondays: List[_dt.date]) -> List[Tuple[int, int]]:
    """Return calendar months touched by the given ISO weeks."""
    months = {
        (day.year, day.month)
        for monday in mondays
        for day in week_days(monday)
    }
    return sorted(months)


###############################################################################
# Raw log access
###############################################################################

def _read_log(data_dir: Path, day: _dt.date, emit: Callable = _echo) -> Optional[str]:
    """Return the raw IRC log text for ``day``.

    Reads ``data_dir/raw/<date>.log`` if present; otherwise attempts to download
    it from gnusha.org and caches it so later runs resume. Returns ``None`` if
    the log cannot be obtained (e.g. no log exists for that day yet).
    """
    raw_dir = data_dir / "raw"
    log_file = raw_dir / f"{day.isoformat()}.log"
    if log_file.exists():
        return read_human_text(log_file)
    try:
        text = gnusha.fetch_log(day)
    except Exception as exc:  # noqa: BLE001
        emit(f"  Could not fetch log for {day}: {exc}")
        return None
    ensure_directory(raw_dir)
    log_file.write_text(text, encoding="utf-8")
    return text


###############################################################################
# Publishing helper + upload manifest
###############################################################################

# Tracks which summary files (by content hash) have been uploaded to which
# remote destination, so resume uploads good-but-never-uploaded files (e.g. one
# generated earlier with --no-upload) without wastefully re-uploading unchanged
# ones. Stored at data_dir/summaries/.upload_state.json.
_UPLOAD_STATE_LOCK = threading.Lock()


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _upload_state_path(data_dir: Path) -> Path:
    return _summaries_dir(data_dir) / ".upload_state.json"


def _read_upload_state(data_dir: Path) -> dict:
    p = _upload_state_path(data_dir)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - corrupt manifest -> treat as empty
            return {}
    return {}


def _is_uploaded(data_dir: Path, remote_key: str, name: str, content_hash: str) -> bool:
    with _UPLOAD_STATE_LOCK:
        return _read_upload_state(data_dir).get(remote_key, {}).get(name) == content_hash


def _mark_uploaded(data_dir: Path, remote_key: str, name: str, content_hash: str) -> None:
    with _UPLOAD_STATE_LOCK:
        state = _read_upload_state(data_dir)
        state.setdefault(remote_key, {})[name] = content_hash
        p = _upload_state_path(data_dir)
        ensure_directory(p.parent)
        p.write_text(json.dumps(state, indent=0, sort_keys=True), encoding="utf-8")


def _publish(
    data_dir: Path,
    content: str,
    output_name: str,
    css_file: str,
    upload: bool,
    remote_user: str,
    remote_host: str,
    remote_path: str,
    prefix: str,
    emit: Callable = _echo,
) -> Optional[Path]:
    """Write markdown, generate HTML, and (idempotently) upload.

    Safe to call on reuse: if uploading is requested but this exact content has
    already been uploaded to this destination, the upload (and HTML regen) is
    skipped. If the content is new/changed OR was never uploaded (e.g. generated
    earlier with --no-upload), it is uploaded and recorded.
    """
    remote_key = f"{remote_user}@{remote_host}:{remote_path}"
    content_hash = _content_hash(content)

    if upload and _is_uploaded(data_dir, remote_key, output_name, content_hash):
        emit(f"  Already uploaded (unchanged): {output_name}")
        return None

    md_file = publishing.output(
        data_dir, content, output_name=output_name, css_file=css_file, upload=upload,
        remote_user=remote_user, remote_host=remote_host, remote_path=remote_path,
        prefix=prefix,
    )
    if md_file:
        html_file = md_file.with_suffix(".html")
        emit(f"  Wrote markdown: {md_file}")
        emit(f"  Wrote HTML:     {html_file}")
        if upload:
            emit(f"  Uploaded to:    {remote_user}@{remote_host}:{remote_path}")
            _mark_uploaded(data_dir, remote_key, output_name, content_hash)
    return md_file


###############################################################################
# Link-following progress output
###############################################################################

def _make_link_progress(emit: Callable = _echo):
    """Return (callback, stats) where the callback streams per-link status via ``emit``.

    ``stats`` accumulates totals so the caller can report an aggregate afterwards.
    """
    stats = {"papers": 0, "pdfs": 0, "failed": 0, "ok": 0, "text_tokens": 0}

    def progress(index: int, total: int, res: dict) -> None:
        url = res.get("url", "")
        title = res.get("title") or ""
        kind = res.get("kind")
        err = res.get("error")
        text = res.get("text") or ""
        scholarly = res.get("scholarly")
        ntok = token_count(text) if text else 0
        stats["text_tokens"] += ntok

        if kind == "pdf":
            stats["pdfs"] += 1
            if scholarly:
                stats["papers"] += 1
            emit(f"  [{index}/{total}] PDF   {url}")
            if title:
                emit(f"             title: {title}")
        elif kind in ("binary", "skipped"):
            emit(f"  [{index}/{total}] skip  {url} ({err or kind})")
        elif err:
            stats["failed"] += 1
            emit(f"  [{index}/{total}] FAIL  {url} — {err}")
            if title:
                emit(f"             title: {title}")
        else:
            stats["ok"] += 1
            label = "PAPER" if scholarly else "link "
            if scholarly:
                stats["papers"] += 1
            emit(f"  [{index}/{total}] {label} {url} ({ntok} tokens)")
            if title:
                emit(f"             title: {title}")

    return progress, stats


###############################################################################
# Daily
###############################################################################

def summarize_day(
    data_dir: Path,
    day: _dt.date,
    model: str = DEFAULT_MODEL,
    fetch_links: bool = True,
    max_links: int = 500,
    max_link_chars: int = 6000,
    force: bool = False,
    upload: bool = True,
    css_file: str = "wrap.css",
    remote_user: str = "bryan",
    remote_host: str = "gnusha.org",
    remote_path: str = DEFAULT_REMOTE_PATH,
    publish: bool = True,
    echo_summary: bool = True,
    emit: Callable = _echo,
) -> Optional[str]:
    """Generate (or reuse) the daily summary for ``day``. Returns markdown text.

    When ``publish`` is set the markdown is uploaded immediately after it is
    written (then the HTML), so each day's file lands on the server as soon as it
    is ready — even when this is one step of a weekly/monthly run. All stdout goes
    through ``emit`` so callers can buffer it when running days concurrently.
    """
    emit(f"=== Daily summary: {day} ===")
    daily_file = _daily_path(data_dir, day)

    if daily_file.exists() and not force:
        content = read_human_text(daily_file)
        if _usable(content):
            emit(f"✓ Daily summary for {day} already done; reusing (use --force to regenerate).")
            emit(f"  Summary file: {daily_file}")
            if publish:
                _publish(data_dir, content, f"daily-{day.isoformat()}", css_file, upload,
                         remote_user, remote_host, remote_path, "daily", emit=emit)
            return content
        emit(f"  Existing daily summary for {day} is empty/corrupt; regenerating.")

    log_text = _read_log(data_dir, day, emit=emit)
    if log_text is None:
        return None
    if not log_text.strip():
        emit(f"  Log for {day} is empty, skipping.")
        return None
    emit(f"  Read log ({len(log_text)} chars, {token_count(log_text)} tokens): "
         f"{LOG_URL_TEMPLATE.format(date=day.isoformat())}")

    link_details = ""
    if fetch_links:
        urls = linkfetch.extract_urls(log_text)
        if urls:
            following = min(len(urls), max_links)
            emit(f"  Following {following} of {len(urls)} posted link(s) "
                 f"(max-links-daily={max_links})…")
            progress, stats = _make_link_progress(emit)
            fetched = linkfetch.fetch_links(
                urls, max_links=max_links, max_chars=max_link_chars, progress=progress,
            )
            link_details = linkfetch.format_link_details(fetched)
            emit(
                f"  Link summary: {stats['ok']} fetched, {stats['pdfs']} PDF(s), "
                f"{stats['papers']} likely paper(s), {stats['failed']} failed; "
                f"{stats['text_tokens']} tokens of link context."
            )
        else:
            emit("  No links posted in this log.")

    prompt = build_daily_summary_prompt(day.isoformat(), log_text, link_details)
    emit(f"  Prompt size: {token_count(prompt)} tokens.")
    emit(f"  Summarizing {day} with {model}…")
    summary = _complete_checked(prompt, model, emit)

    if not _usable(summary):
        emit(f"  ✗ Model returned no usable summary for {day} after {LLM_RETRIES} "
             f"attempts; NOT writing/uploading (leaving any existing file intact).")
        return _existing_if_usable(daily_file)

    ensure_directory(daily_file.parent)
    daily_file.write_text(summary, encoding="utf-8")
    emit(f"  Saved daily summary ({token_count(summary)} tokens): {daily_file}")

    if publish:
        _publish(data_dir, summary, f"daily-{day.isoformat()}", css_file, upload,
                 remote_user, remote_host, remote_path, "daily", emit=emit)
    if echo_summary:
        emit("\n" + "=" * 70 + f"\nDaily summary for {day}:\n" + "=" * 70 + "\n")
        emit(summary + "\n")
    return summary


def summarize_days(
    data_dir: Path,
    start: _dt.date,
    end: _dt.date,
    concurrency: int = DEFAULT_CONCURRENCY,
    **kwargs,
) -> None:
    """Summarize each day in the inclusive range ``start``..``end``.

    Up to ``concurrency`` days are summarized in parallel; each day's log output
    is buffered and printed as one atomic block so concurrent output stays readable.
    """
    days = list(daterange(start, end))
    workers = max(1, min(concurrency, len(days)))
    if len(days) > 1:
        _echo(f"=== Summarizing {len(days)} day(s) {start} … {end} "
              f"(concurrency={workers}) ===")
    jobs = [
        (day, (lambda emit, d=day: summarize_day(data_dir, d, emit=emit, **kwargs)))
        for day in days
    ]
    _run_day_jobs(jobs, concurrency)


def _raw_log_days(data_dir: Path) -> List[_dt.date]:
    """Return all dates with local raw IRC logs available."""
    raw_dir = data_dir / "raw"
    if not raw_dir.exists():
        return []

    days: List[_dt.date] = []
    for path in raw_dir.glob("*.log"):
        try:
            days.append(_dt.date.fromisoformat(path.stem))
        except ValueError:
            continue
    return sorted(days)


def summarize_missing_days(
    data_dir: Path,
    start: Optional[_dt.date] = None,
    end: Optional[_dt.date] = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    force: bool = False,
    **kwargs,
) -> None:
    """Summarize all raw-log days that do not have a usable daily summary yet.

    Discovery is based on local ``data_dir/raw/YYYY-MM-DD.log`` files. A daily
    summary is considered done only when its canonical cached markdown exists and
    passes the same usability threshold used by ``summarize_day`` resume logic.
    With ``force=True``, every matching raw-log day is regenerated.
    """
    days = _raw_log_days(data_dir)
    if start is not None:
        days = [day for day in days if day >= start]
    if end is not None:
        days = [day for day in days if day <= end]

    if not days:
        raw_dir = data_dir / "raw"
        _echo(f"No raw log files found in {raw_dir}. Run download first, or pass --data-dir.")
        return

    rebuild_count = sum(
        1 for day in days
        if force or _existing_if_usable(_daily_path(data_dir, day)) is None
    )
    reuse_count = len(days) - rebuild_count

    workers = max(1, min(concurrency, len(days)))
    action = "Regenerating" if force else "Summarizing missing"
    bounds = f"{days[0]} … {days[-1]}" if len(days) > 1 else str(days[0])
    _echo(f"=== {action} {rebuild_count} daily summaries ({bounds}; "
          f"checking={len(days)}; concurrency={workers}; reusing={reuse_count}) ===")

    jobs = [
        (day, (lambda emit, d=day: summarize_day(data_dir, d, force=force, emit=emit, **kwargs)))
        for day in days
    ]
    _run_day_jobs(jobs, concurrency, buffer_output=False)


###############################################################################
# Weekly
###############################################################################

def _ensure_daily_summaries(
    data_dir: Path,
    days: List[_dt.date],
    auto: bool,
    model: str,
    fetch_links: bool,
    max_links: int,
    max_link_chars: int,
    force: bool,
    upload: bool,
    css_file: str,
    remote_user: str,
    remote_host: str,
    remote_path: str,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> Tuple[List[Tuple[_dt.date, str]], int]:
    """Collect daily summaries for ``days``, generating missing ones if ``auto``.

    Missing/empty dailies are generated in parallel (up to ``concurrency``) and each
    is published/uploaded immediately (respecting ``upload``) so it lands on the
    server as soon as it is ready. Good cached summaries are reused as-is (not
    re-uploaded). Returns ``(collected, generated_count)`` where ``generated_count``
    is how many dailies were actually (re)written this call — used to decide whether
    the parent weekly summary is now stale and must be rebuilt.
    """
    def _cached(day: _dt.date) -> Optional[str]:
        """Return a cached daily summary only if it is non-empty/usable."""
        f = _daily_path(data_dir, day)
        if not f.exists():
            return None
        text = read_human_text(f)
        return text if len(text.strip()) >= MIN_SUMMARY_CHARS else None

    reuse: dict = {}
    to_generate: List[_dt.date] = []
    for day in days:
        cached = None if force else _cached(day)
        if cached is not None:
            reuse[day] = cached
        elif not auto:
            if _daily_path(data_dir, day).exists():
                _echo(f"  (existing daily summary for {day} is empty; use --auto to regenerate)")
            else:
                _echo(f"  (no daily summary for {day}; use --auto to generate)")
        else:
            to_generate.append(day)

    generated: dict = {}
    if to_generate:
        jobs = [
            (day, (lambda emit, d=day: summarize_day(
                data_dir, d, model=model, fetch_links=fetch_links, max_links=max_links,
                max_link_chars=max_link_chars, force=force, upload=upload, css_file=css_file,
                remote_user=remote_user, remote_host=remote_host, remote_path=remote_path,
                publish=True, echo_summary=False, emit=emit,
            )))
            for day in to_generate
        ]
        generated = _run_day_jobs(jobs, concurrency)

    collected: List[Tuple[_dt.date, str]] = []
    for day in days:
        text = reuse.get(day) if day in reuse else generated.get(day)
        if text:
            collected.append((day, text))

    # Idempotently ensure reused (unchanged) dailies are on the server, in
    # parallel. _publish skips any already-uploaded content, so on a steady-state
    # resume these are quick no-ops; only never-uploaded files actually upload.
    # (Skipped entirely with --no-upload: the local files already exist.)
    if upload and reuse:
        pub_jobs = [
            (day, (lambda emit, d=day: _publish(
                data_dir, reuse[d], f"daily-{d.isoformat()}", css_file, upload,
                remote_user, remote_host, remote_path, "daily", emit=emit) and None))
            for day in days if day in reuse
        ]
        _run_day_jobs(pub_jobs, concurrency)

    return collected, len(generated)


def summarize_week(
    data_dir: Path,
    any_day: _dt.date,
    model: str = DEFAULT_MODEL,
    auto: bool = True,
    fetch_links: bool = True,
    max_links: int = 500,
    max_link_chars: int = 6000,
    force: bool = False,
    force_daily: bool = False,
    upload: bool = True,
    css_file: str = "wrap.css",
    remote_user: str = "bryan",
    remote_host: str = "gnusha.org",
    remote_path: str = DEFAULT_REMOTE_PATH,
    publish: bool = True,
    echo_summary: bool = True,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> Optional[str]:
    """Generate the weekly summary for the ISO week containing ``any_day``."""
    summary, _changed = _do_week(
        data_dir, any_day, model=model, auto=auto, fetch_links=fetch_links,
        max_links=max_links, max_link_chars=max_link_chars, force=force,
        force_daily=force_daily, upload=upload, css_file=css_file, remote_user=remote_user,
        remote_host=remote_host, remote_path=remote_path, publish=publish,
        echo_summary=echo_summary, concurrency=concurrency,
    )
    return summary


def _do_week(
    data_dir: Path,
    any_day: _dt.date,
    model: str,
    auto: bool,
    fetch_links: bool,
    max_links: int,
    max_link_chars: int,
    force: bool,
    force_daily: bool,
    upload: bool,
    css_file: str,
    remote_user: str,
    remote_host: str,
    remote_path: str,
    publish: bool,
    echo_summary: bool,
    concurrency: int,
) -> Tuple[Optional[str], bool]:
    """Generate/reuse a single weekly summary (fills daily gaps first).

    Used by the standalone ``summarize-week`` command. Returns ``(summary, changed)``.
    """
    monday = monday_of(any_day)
    label = week_label(monday)
    days = week_days(monday)
    _echo(f"=== Weekly summary: {label} ({days[0]} … {days[-1]}) ===")

    daily, _gen_count = _ensure_daily_summaries(
        data_dir, days, auto, model, fetch_links, max_links, max_link_chars, force_daily,
        upload=upload, css_file=css_file, remote_user=remote_user,
        remote_host=remote_host, remote_path=remote_path, concurrency=concurrency,
    )
    return _consolidate_week(
        data_dir, monday, daily, model=model, force=force, upload=upload, css_file=css_file,
        remote_user=remote_user, remote_host=remote_host, remote_path=remote_path,
        publish=publish, echo_summary=echo_summary, emit=_echo,
    )


def _consolidate_week(
    data_dir: Path,
    monday: _dt.date,
    daily: List[Tuple[_dt.date, str]],
    model: str,
    force: bool,
    upload: bool,
    css_file: str,
    remote_user: str,
    remote_host: str,
    remote_path: str,
    publish: bool,
    echo_summary: bool,
    emit: Callable = _echo,
) -> Tuple[Optional[str], bool]:
    """Consolidate already-generated dailies into the weekly summary.

    The dailies are passed in (generated elsewhere), so this only decides whether
    the weekly is stale and, if so, rebuilds it. Resume-aware: the weekly is
    rebuilt only if it is missing/empty, ``force`` is set, or an underlying daily
    file is newer than the weekly (``_is_stale``). ``changed`` reports whether the
    weekly was (re)written. All output goes through ``emit`` so this can run inside
    a parallel worker (buffered) or live.
    """
    label = week_label(monday)
    days = week_days(monday)
    weekly_file = _weekly_path(data_dir, monday)
    existing = _existing_if_usable(weekly_file)
    if existing is None and weekly_file.exists():
        emit(f"  Existing weekly summary {label} is empty/corrupt; will regenerate.")

    if not daily:
        emit(f"  No daily summaries available for {label}; nothing to do.")
        return existing, False

    stale = _is_stale(weekly_file, [_daily_path(data_dir, d) for d in days])
    needs_rebuild = force or existing is None or stale
    if not needs_rebuild:
        emit(f"✓ Weekly {label} up to date ({len(daily)} days, no changes); reusing.")
        emit(f"  Summary file: {weekly_file}")
        if publish:
            _publish(data_dir, existing, f"weekly-{label}", css_file, upload,
                     remote_user, remote_host, remote_path, "weekly", emit=emit)
        return existing, False

    if stale and existing is not None:
        emit(f"  A daily summary is newer than weekly {label}; rebuilding.")

    combined = "\n\n".join(
        f"## Daily summary for {day.isoformat()}\n\n{text}" for day, text in daily
    )
    prompt = build_weekly_summary_prompt(label, combined)
    emit(f"  Consolidating {len(daily)} daily summaries into weekly {label} "
         f"({token_count(prompt)} prompt tokens) with {model}…")
    summary = _complete_checked(prompt, model, emit)

    if not _usable(summary):
        emit(f"  ✗ Model returned no usable weekly summary for {label} after "
             f"{LLM_RETRIES} attempts; NOT writing/uploading (leaving any existing "
             f"file intact).")
        return existing, False

    ensure_directory(weekly_file.parent)
    weekly_file.write_text(summary, encoding="utf-8")
    emit(f"  Saved weekly summary ({token_count(summary)} tokens): {weekly_file}")

    if publish:
        _publish(data_dir, summary, f"weekly-{label}", css_file, upload,
                 remote_user, remote_host, remote_path, "weekly", emit=emit)
    if echo_summary:
        emit("\n" + "=" * 70 + f"\nWeekly summary for {label}:\n" + "=" * 70 + "\n")
        emit(summary + "\n")
    return summary, True


def summarize_weeks(
    data_dir: Path,
    start: Optional[_dt.date] = None,
    end: Optional[_dt.date] = None,
    model: str = DEFAULT_MODEL,
    force: bool = False,
    upload: bool = True,
    css_file: str = "wrap.css",
    remote_user: str = "bryan",
    remote_host: str = "gnusha.org",
    remote_path: str = DEFAULT_REMOTE_PATH,
    publish: bool = True,
    echo_summary: bool = True,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> None:
    """Summarize weeks discovered from existing usable daily summaries.

    Discovery is based on ``data_dir/summaries/daily/YYYY-MM-DD.md`` files. This
    intentionally does not generate missing dailies; use ``summarize-days`` for
    that. Resume skips weekly summaries that are present and newer than all
    available daily summaries in that week unless ``force`` is set.
    """
    days = _daily_summary_days(data_dir)
    if start is not None:
        days = [day for day in days if day >= start]
    if end is not None:
        days = [day for day in days if day <= end]

    if not days:
        daily_dir = _summaries_dir(data_dir) / "daily"
        _echo(f"No usable daily summaries found in {daily_dir}.")
        return

    daily_by_week: dict = {}
    for day in days:
        daily_by_week.setdefault(monday_of(day), []).append(day)

    candidates = [monday for monday, _wk_days in sorted(daily_by_week.items())]
    rebuild_count = 0
    for monday, wk_days in sorted(daily_by_week.items()):
        weekly_file = _weekly_path(data_dir, monday)
        existing = _existing_if_usable(weekly_file)
        stale = _is_stale(weekly_file, [_daily_path(data_dir, d) for d in wk_days])
        if force or existing is None or stale:
            rebuild_count += 1

    reuse_count = len(candidates) - rebuild_count
    workers = max(1, min(concurrency, len(candidates)))
    action = "Regenerating" if force else "Summarizing missing/stale"
    _echo(f"=== {action} {rebuild_count} weekly summaries from existing daily "
          f"summaries (checking={len(candidates)}; concurrency={workers}; "
          f"reusing={reuse_count}) ===")
    _echo("  Launching weekly jobs: " + ", ".join(week_label(m) for m in candidates))

    def _week_job(monday: _dt.date) -> Callable:
        def run(emit: Callable):
            wk_days = daily_by_week[monday]
            daily = [
                (day, _existing_if_usable(_daily_path(data_dir, day)) or "")
                for day in wk_days
            ]
            daily = [(day, text) for day, text in daily if text]
            emit(f"=== Weekly summary: {week_label(monday)} "
                 f"({wk_days[0]} … {wk_days[-1]}, {len(daily)} daily summaries) ===")
            return _consolidate_week(
                data_dir, monday, daily, model=model, force=force, upload=upload,
                css_file=css_file, remote_user=remote_user, remote_host=remote_host,
                remote_path=remote_path, publish=publish, echo_summary=echo_summary, emit=emit,
            )
        return run

    _run_day_jobs(
        [(monday, _week_job(monday)) for monday in candidates],
        concurrency,
        buffer_output=False,
    )


###############################################################################
# Monthly
###############################################################################

def summarize_month(
    data_dir: Path,
    year: int,
    month: int,
    model: str = DEFAULT_MODEL,
    auto: bool = True,
    fetch_links: bool = True,
    max_links: int = 500,
    max_link_chars: int = 6000,
    force: bool = False,
    force_weekly: bool = False,
    force_daily: bool = False,
    upload: bool = True,
    css_file: str = "wrap.css",
    remote_user: str = "bryan",
    remote_host: str = "gnusha.org",
    remote_path: str = DEFAULT_REMOTE_PATH,
    publish: bool = True,
    echo_summary: bool = True,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> Optional[str]:
    """Generate the monthly summary for ``year``-``month`` from its weekly summaries."""
    label = f"{year:04d}-{month:02d}"
    mondays = month_weeks(year, month)
    _echo(f"=== Monthly summary: {label} ({len(mondays)} overlapping weeks) ===")

    # Phase 1: generate/refresh EVERY daily across the whole month in one parallel
    # pool (up to ``concurrency`` at once), rather than a separate ~7-day batch per
    # week. This is the bulk of the work, so month-wide parallelism is the big win.
    all_days: List[_dt.date] = []
    for monday in mondays:
        all_days.extend(week_days(monday))
    _echo(f"  Phase 1/3: generating/refreshing {len(all_days)} daily summaries "
          f"across {len(mondays)} weeks (concurrency={min(concurrency, len(all_days))})…")
    daily_list, _gen = _ensure_daily_summaries(
        data_dir, all_days, auto, model, fetch_links, max_links, max_link_chars, force_daily,
        upload=upload, css_file=css_file, remote_user=remote_user,
        remote_host=remote_host, remote_path=remote_path, concurrency=concurrency,
    )
    daily_by_date = dict(daily_list)

    # Phase 2: consolidate the weeks IN PARALLEL — each week only depends on its own
    # (now-complete) dailies. Staleness (a daily newer than the weekly) triggers a
    # rebuild, so a daily regenerated in phase 1 correctly rebuilds its weekly.
    _echo(f"  Phase 2/3: consolidating {len(mondays)} weekly summaries "
          f"(concurrency={min(concurrency, len(mondays))})…")
    _echo("  Launching weekly jobs: " + ", ".join(week_label(m) for m in mondays))

    def _week_job(monday: _dt.date) -> Callable:
        def run(emit: Callable):
            wk_days = week_days(monday)
            wk_daily = [(d, daily_by_date[d]) for d in wk_days if d in daily_by_date]
            emit(f"=== Weekly summary: {week_label(monday)} "
                 f"({wk_days[0]} … {wk_days[-1]}) ===")
            return _consolidate_week(
                data_dir, monday, wk_daily, model=model, force=force_weekly, upload=upload,
                css_file=css_file, remote_user=remote_user, remote_host=remote_host,
                remote_path=remote_path, publish=True, echo_summary=False, emit=emit,
            )
        return run

    week_results = _run_day_jobs(
        [(m, _week_job(m)) for m in mondays],
        concurrency,
        buffer_output=False,
    )

    weekly_texts: List[Tuple[str, str]] = []
    any_week_changed = False
    for monday in mondays:
        res = week_results.get(monday)
        if not res:
            continue
        wk_summary, wk_changed = res
        any_week_changed = any_week_changed or wk_changed
        if _usable(wk_summary):
            weekly_texts.append((week_label(monday), wk_summary))

    _echo("  Phase 3/3: consolidating the monthly summary…")
    summary, _changed = _consolidate_month(
        data_dir, year, month, mondays, weekly_texts, model=model, force=force,
        upstream_changed=any_week_changed, upload=upload, css_file=css_file,
        remote_user=remote_user, remote_host=remote_host, remote_path=remote_path,
        publish=publish, echo_summary=echo_summary, emit=_echo,
    )
    return summary


def _consolidate_month(
    data_dir: Path,
    year: int,
    month: int,
    mondays: List[_dt.date],
    weekly_texts: List[Tuple[str, str]],
    model: str,
    force: bool,
    upstream_changed: bool,
    upload: bool,
    css_file: str,
    remote_user: str,
    remote_host: str,
    remote_path: str,
    publish: bool,
    echo_summary: bool,
    emit: Callable = _echo,
) -> Tuple[Optional[str], bool]:
    """Consolidate already-generated weeklies into one monthly summary."""
    label = f"{year:04d}-{month:02d}"
    monthly_file = _monthly_path(data_dir, year, month)
    existing = _existing_if_usable(monthly_file)
    if existing is None and monthly_file.exists():
        emit(f"  Existing monthly summary {label} is empty/corrupt; will regenerate.")

    if not weekly_texts:
        emit(f"  No weekly summaries available for {label}; nothing to do.")
        return existing, False

    stale = _is_stale(monthly_file, [_weekly_path(data_dir, m) for m in mondays])
    needs_rebuild = force or existing is None or upstream_changed or stale
    if not needs_rebuild:
        emit(f"✓ Monthly {label} up to date (no weeks changed); reusing.")
        emit(f"  Summary file: {monthly_file}")
        if publish:
            _publish(data_dir, existing, f"monthly-{label}", css_file, upload,
                     remote_user, remote_host, remote_path, "monthly", emit=emit)
        return existing, False

    if upstream_changed and existing is not None:
        emit(f"  Underlying week(s) changed; rebuilding monthly {label}.")
    elif stale and existing is not None:
        emit(f"  A weekly summary is newer than monthly {label}; rebuilding.")

    combined = "\n\n".join(
        f"## Weekly summary: {wk}\n\n{text}" for wk, text in weekly_texts
    )
    prompt = build_monthly_summary_prompt(label, combined)
    emit(f"  Consolidating {len(weekly_texts)} weekly summaries into monthly {label} "
         f"({token_count(prompt)} prompt tokens) with {model}…")
    summary = _complete_checked(prompt, model, emit)

    if not _usable(summary):
        emit(f"  ✗ Model returned no usable monthly summary for {label} after "
             f"{LLM_RETRIES} attempts; NOT writing/uploading (leaving any existing "
             f"file intact).")
        return existing, False

    ensure_directory(monthly_file.parent)
    monthly_file.write_text(summary, encoding="utf-8")
    emit(f"  Saved monthly summary ({token_count(summary)} tokens): {monthly_file}")

    if publish:
        _publish(data_dir, summary, f"monthly-{label}", css_file, upload,
                 remote_user, remote_host, remote_path, "monthly", emit=emit)
    if echo_summary:
        emit("\n" + "=" * 70 + f"\nMonthly summary for {label}:\n" + "=" * 70 + "\n")
        emit(summary + "\n")
    return summary, True


def summarize_months(
    data_dir: Path,
    start: Optional[_dt.date] = None,
    end: Optional[_dt.date] = None,
    model: str = DEFAULT_MODEL,
    force: bool = False,
    upload: bool = True,
    css_file: str = "wrap.css",
    remote_user: str = "bryan",
    remote_host: str = "gnusha.org",
    remote_path: str = DEFAULT_REMOTE_PATH,
    publish: bool = True,
    echo_summary: bool = True,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> None:
    """Summarize months discovered from existing usable weekly summaries.

    Discovery is based on ``data_dir/summaries/weekly/YYYY-Www.md`` files. This
    intentionally does not generate missing weeklies or dailies; run
    ``summarize-weeks`` first when needed.
    """
    all_mondays = _weekly_summary_mondays(data_dir)
    if not all_mondays:
        weekly_dir = _summaries_dir(data_dir) / "weekly"
        _echo(f"No usable weekly summaries found in {weekly_dir}.")
        return

    months = _months_overlapped_by_weeks(all_mondays)
    if start is not None:
        months = [(y, m) for y, m in months if _dt.date(y, m, 1) >= start]
    if end is not None:
        months = [(y, m) for y, m in months if _dt.date(y, m, 1) <= end]

    if not months:
        _echo("No months matched the requested range.")
        return

    weekly_set = set(all_mondays)
    candidates: List[_dt.date] = []
    rebuild_count = 0
    month_mondays: dict = {}
    for year, month in months:
        key = _dt.date(year, month, 1)
        mondays = [m for m in month_weeks(year, month) if m in weekly_set]
        month_mondays[key] = mondays
        candidates.append(key)
        monthly_file = _monthly_path(data_dir, year, month)
        existing = _existing_if_usable(monthly_file)
        stale = _is_stale(monthly_file, [_weekly_path(data_dir, m) for m in mondays])
        if force or existing is None or stale:
            rebuild_count += 1

    reuse_count = len(candidates) - rebuild_count
    workers = max(1, min(concurrency, len(candidates)))
    action = "Regenerating" if force else "Summarizing missing/stale"
    _echo(f"=== {action} {rebuild_count} monthly summaries from existing weekly "
          f"summaries (checking={len(candidates)}; concurrency={workers}; "
          f"reusing={reuse_count}) ===")
    _echo("  Launching monthly jobs: " + ", ".join(f"{d.year:04d}-{d.month:02d}" for d in candidates))

    def _month_job(first_day: _dt.date) -> Callable:
        def run(emit: Callable):
            year, month = first_day.year, first_day.month
            mondays = month_mondays[first_day]
            weekly_texts = []
            for monday in mondays:
                text = _existing_if_usable(_weekly_path(data_dir, monday))
                if text:
                    weekly_texts.append((week_label(monday), text))
            emit(f"=== Monthly summary: {year:04d}-{month:02d} "
                 f"({len(weekly_texts)} weekly summaries) ===")
            return _consolidate_month(
                data_dir, year, month, mondays, weekly_texts, model=model,
                force=force, upstream_changed=False, upload=upload,
                css_file=css_file, remote_user=remote_user, remote_host=remote_host,
                remote_path=remote_path, publish=publish, echo_summary=echo_summary, emit=emit,
            )
        return run

    _run_day_jobs(
        [(first_day, _month_job(first_day)) for first_day in candidates],
        concurrency,
        buffer_output=False,
    )
