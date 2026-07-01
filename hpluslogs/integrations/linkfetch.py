"""Fetch and extract readable text from links posted in the chat logs.

The IRC logs are plain text, but the *links* people post (papers, preprints,
blog posts) point at HTML/PDF pages. To let the summarizer describe those
manuscripts accurately we follow each unique link and pull out a title and a
snippet of readable text. HTML is parsed with lightweight regex/`html`-stdlib
extraction (no BeautifulSoup dependency); PDFs and other binaries are recorded
by URL/title only.
"""

from __future__ import annotations

import html as _html
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests

try:  # lxml (libxml2) is the preferred fallback HTML parser when mdream is absent.
    from lxml import html as _lxml_html
except Exception:  # noqa: BLE001 - lxml is optional; regex is the last resort.
    _lxml_html = None

# A permissive URL matcher. IRC lines frequently wrap URLs in <> or () or end a
# sentence with them, so we strip trailing punctuation afterwards (respecting
# balanced parens/brackets, since e.g. cell.com DOIs contain "(24)").
_URL_RE = re.compile(r"https?://[^\s<>\"'`]+", re.IGNORECASE)

# Sentence punctuation that is never part of a URL.
_TRAILING_PUNCT = ".,;:!?\"'>"


def _strip_trailing(url: str) -> str:
    """Strip sentence punctuation and unbalanced closing brackets from a URL."""
    changed = True
    while changed and url:
        changed = False
        while url and url[-1] in _TRAILING_PUNCT:
            url = url[:-1]
            changed = True
        if url and url[-1] in ")]":
            opener = "(" if url[-1] == ")" else "["
            if url.count(opener) < url.count(url[-1]):
                url = url[:-1]
                changed = True
    return url

# Extensions we treat as non-HTML binaries / media (not worth HTML extraction).
_BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".ico",
    ".mp4", ".webm", ".mov", ".avi", ".mkv", ".mp3", ".wav", ".ogg",
    ".zip", ".tar", ".gz", ".xz", ".bz2", ".7z", ".rar",
    ".exe", ".dmg", ".iso", ".bin",
}

# Media/junk hosts we don't bother following (still counted, just skipped).
_SKIP_HOSTS = {
    "quassel-irc.org",
}

# Hosts / path hints that usually indicate a scholarly manuscript or preprint.
# Used only to label progress output ("PAPER"); the LLM still makes the final call.
_SCHOLARLY_HINTS = (
    "arxiv.org", "biorxiv.org", "medrxiv.org", "chemrxiv.org", "osf.io",
    "doi.org", "nature.com", "science.org", "sciencedirect.com", "cell.com",
    "pnas.org", "ncbi.nlm.nih.gov", "pubmed", "pmc", "wiley.com", "springer",
    "link.springer.com", "frontiersin.org", "journals.plos.org", "plos.org",
    "mdpi.com", "biomedcentral.com", "elifesciences.org", "jamanetwork.com",
    "thelancet.com", "nejm.org", "oup.com", "academic.oup.com", "tandfonline.com",
    "researchgate.net", "semanticscholar.org", "ssrn.com", "eprint.iacr.org",
    "acm.org", "ieee.org", "ieeexplore", "aeb", "acs.org", "rsc.org",
    "ntrs.nasa.gov", "europepmc.org",
)


def looks_scholarly(url: str) -> bool:
    """Heuristic: does this URL look like a manuscript/preprint/journal article?"""
    low = url.lower()
    if low.endswith(".pdf"):
        return True
    return any(hint in low for hint in _SCHOLARLY_HINTS)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36 hpluslogs-summarizer/1.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/pdf,*/*",
}

_SCRIPT_STYLE_RE = re.compile(r"<(script|style|noscript|template)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_META_DESC_RE = re.compile(
    r"""<meta[^>]+(?:name|property)=["'](?:description|og:description|citation_abstract)["'][^>]*>""",
    re.IGNORECASE,
)
_META_CONTENT_RE = re.compile(r"""content=["'](.*?)["']""", re.IGNORECASE | re.DOTALL)
_CITATION_TITLE_RE = re.compile(
    r"""<meta[^>]+(?:name|property)=["'](?:citation_title|og:title|dc.title)["'][^>]*content=["'](.*?)["']""",
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_MULTINEWLINE_RE = re.compile(r"\n\s*\n\s*\n+")


# ---------------------------------------------------------------------------
# HTML -> Markdown via mdream (https://github.com/harlan-zw/mdream)
#
# mdream produces LLM-optimized ("minimal" GFM) markdown that is ~2x fewer
# tokens than naive HTML text extraction, cutting summarization cost. It is a
# Node CLI that reads HTML from stdin. We locate the locally-installed binary
# (hpluslogs/node_modules/.bin/mdream) or one on PATH, and fall back to a plain
# regex text-extraction if mdream is unavailable or errors.
# ---------------------------------------------------------------------------

_MDREAM_CMD: Optional[List[str]] = None  # cached: command list, or [] if unavailable


def _mdream_cmd() -> List[str]:
    """Return the command prefix to invoke mdream, or [] if not available."""
    global _MDREAM_CMD
    if _MDREAM_CMD is not None:
        return _MDREAM_CMD

    # Allow an explicit override.
    override = os.environ.get("MDREAM_BIN")
    candidates: List[Path] = []
    if override:
        candidates.append(Path(override))
    # Locally installed alongside the package (hpluslogs/node_modules/.bin/mdream).
    pkg_root = Path(__file__).resolve().parent.parent
    candidates.append(pkg_root / "node_modules" / ".bin" / "mdream")

    for cand in candidates:
        if cand.exists():
            _MDREAM_CMD = [str(cand)]
            return _MDREAM_CMD

    on_path = shutil.which("mdream")
    if on_path:
        _MDREAM_CMD = [on_path]
        return _MDREAM_CMD

    _MDREAM_CMD = []
    return _MDREAM_CMD


def _run_mdream(raw_html: str, url: str, timeout: int = 30) -> Optional[str]:
    """Convert HTML to markdown with mdream; return None on any failure."""
    cmd = _mdream_cmd()
    if not cmd:
        return None
    try:
        proc = subprocess.run(
            cmd + ["--preset", "minimal", "--origin", url],
            input=raw_html, capture_output=True, text=True, timeout=timeout,
        )
    except Exception:  # noqa: BLE001 - node missing, timeout, etc.
        return None
    if proc.returncode == 0 and proc.stdout and proc.stdout.strip():
        return proc.stdout
    return None


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + " …[truncated]"
    return text


def _html_to_markdown(raw_html: str, url: str, max_chars: int) -> str:
    """Prefer mdream markdown; fall back to lxml then regex text extraction."""
    md = _run_mdream(raw_html, url)
    if md is not None:
        return _truncate(md, max_chars)
    return _clean_text(raw_html, max_chars)


def extract_urls(text: str) -> List[str]:
    """Extract unique http(s) URLs from ``text`` preserving first-seen order."""
    seen: Dict[str, None] = {}
    for match in _URL_RE.finditer(text):
        url = _strip_trailing(match.group(0))
        if url and url not in seen:
            seen[url] = None
    return list(seen.keys())


def _looks_binary(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in _BINARY_EXTS)


def _is_pdf(url: str, content_type: str) -> bool:
    if "application/pdf" in content_type.lower():
        return True
    return urlparse(url).path.lower().endswith(".pdf")


def _title_from_url(url: str) -> str:
    """Derive a human-ish title from a URL path when no HTML title is available."""
    parsed = urlparse(url)
    tail = parsed.path.rstrip("/").split("/")[-1] or parsed.netloc
    tail = re.sub(r"\.[a-z0-9]{1,5}$", "", tail, flags=re.IGNORECASE)
    tail = tail.replace("-", " ").replace("_", " ").strip()
    return tail or parsed.netloc


# Non-content elements to strip before extracting readable text.
_LXML_DROP_XPATH = (
    "//script|//style|//noscript|//template|//svg|//form|//nav|//footer"
    "|//header|//aside|//button|//iframe"
)


def _lxml_extract(raw_html: str, max_chars: int) -> Optional[str]:
    """Extract readable text using lxml (libxml2). Returns None if unavailable."""
    if _lxml_html is None:
        return None
    try:
        doc = _lxml_html.fromstring(raw_html)
    except Exception:  # noqa: BLE001 - malformed markup, empty doc, etc.
        return None
    for el in doc.xpath(_LXML_DROP_XPATH):
        try:
            el.drop_tree()
        except Exception:  # noqa: BLE001
            pass
    # Prefer the main article body when the page marks one up.
    main = doc.xpath("//article") or doc.xpath("//main") or doc.xpath("//*[@role='main']")
    node = main[0] if main else doc
    # Join text nodes with spaces so adjacent block elements don't mash together.
    text = " ".join(t for t in node.itertext())
    text = _html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = _MULTINEWLINE_RE.sub("\n\n", text).strip()
    if not text:
        return None
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + " …[truncated]"
    return text


def _regex_text(raw_html: str, max_chars: int) -> str:
    """Last-resort text extraction with regex (used only if lxml is unavailable)."""
    body = _SCRIPT_STYLE_RE.sub(" ", raw_html)
    body = _TAG_RE.sub(" ", body)
    body = _html.unescape(body)
    body = _WS_RE.sub(" ", body)
    body = _MULTINEWLINE_RE.sub("\n\n", body)
    body = "\n".join(line.strip() for line in body.splitlines())
    body = _MULTINEWLINE_RE.sub("\n\n", body).strip()
    if len(body) > max_chars:
        body = body[:max_chars].rsplit(" ", 1)[0] + " …[truncated]"
    return body


def _clean_text(raw_html: str, max_chars: int) -> str:
    """Extract readable text: lxml (libxml2) preferred, regex as last resort."""
    text = _lxml_extract(raw_html, max_chars)
    if text:
        return text
    return _regex_text(raw_html, max_chars)


def _lxml_meta(raw_html: str) -> Dict[str, str]:
    """Extract title/description via lxml. Returns {} if lxml is unavailable."""
    if _lxml_html is None:
        return {}
    try:
        doc = _lxml_html.fromstring(raw_html)
    except Exception:  # noqa: BLE001
        return {}
    meta: Dict[str, str] = {}
    titles = doc.xpath("//title/text()")
    if titles:
        meta["title"] = _html.unescape(titles[0]).strip()
    if not meta.get("title"):
        for name in ("citation_title", "og:title", "dc.title"):
            vals = doc.xpath(
                "//meta[@name=$n or @property=$n]/@content", n=name
            )
            if vals and vals[0].strip():
                meta["title"] = _html.unescape(vals[0]).strip()
                break
    for name in ("description", "og:description", "citation_abstract"):
        vals = doc.xpath("//meta[@name=$n or @property=$n]/@content", n=name)
        if vals and vals[0].strip():
            meta["description"] = _html.unescape(vals[0]).strip()
            break
    return meta


def _extract_meta(raw_html: str) -> Dict[str, str]:
    """Extract title/description, preferring lxml with a regex fallback."""
    meta = _lxml_meta(raw_html)
    if meta.get("title") or meta.get("description"):
        return meta
    title_match = _TITLE_RE.search(raw_html)
    if title_match:
        meta["title"] = _html.unescape(_TAG_RE.sub("", title_match.group(1))).strip()
    cite_title = _CITATION_TITLE_RE.search(raw_html)
    if cite_title and not meta.get("title"):
        meta["title"] = _html.unescape(cite_title.group(1)).strip()
    desc_tag = _META_DESC_RE.search(raw_html)
    if desc_tag:
        content = _META_CONTENT_RE.search(desc_tag.group(0))
        if content:
            meta["description"] = _html.unescape(content.group(1)).strip()
    return meta


def fetch_url(url: str, max_chars: int = 6000, timeout: int = 20) -> Dict[str, Optional[str]]:
    """Fetch a single URL and return a dict with title/text/kind/error.

    Never raises: failures are reported in the ``error`` field so a single dead
    link cannot abort a whole day's summary.
    """
    result: Dict[str, Optional[str]] = {
        "url": url, "title": None, "text": None, "kind": "html", "error": None,
        "scholarly": looks_scholarly(url),
    }

    host = urlparse(url).netloc.lower()
    if any(host == h or host.endswith("." + h) for h in _SKIP_HOSTS):
        result["kind"] = "skipped"
        result["error"] = "skipped host"
        return result

    if _looks_binary(url):
        result["kind"] = "binary"
        result["title"] = _title_from_url(url)
        result["error"] = "binary/media (not fetched)"
        return result

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout, allow_redirects=True, stream=True)
        content_type = resp.headers.get("Content-Type", "")
        if _is_pdf(url, content_type):
            result["kind"] = "pdf"
            result["title"] = _title_from_url(url)
            result["error"] = None
            resp.close()
            return result
        # Determine the real encoding. requests defaults resp.encoding to
        # ISO-8859-1 for text/* responses that lack a charset in the header,
        # which mojibakes UTF-8 pages (e.g. smart quotes -> "âAMCHEPRYâ"). When
        # the header has no charset, prefer the content-sniffed encoding.
        if "charset=" in content_type.lower():
            enc = resp.encoding or "utf-8"
        else:
            enc = resp.apparent_encoding or resp.encoding or "utf-8"
        # Read a bounded amount of the body to avoid huge pages. We keep more
        # raw HTML than the target text budget because mdream/extraction will
        # collapse it substantially.
        raw = resp.content[: max_chars * 20].decode(enc, errors="replace")
        resp.close()
        if resp.status_code >= 400:
            result["error"] = f"HTTP {resp.status_code}"
            result["title"] = _title_from_url(url)
            return result
        meta = _extract_meta(raw)
        result["title"] = meta.get("title") or _title_from_url(url)
        text = _html_to_markdown(raw, url, max_chars)
        if meta.get("description") and meta["description"] not in text:
            text = "Description: " + meta["description"] + "\n\n" + text
        result["text"] = text
    except Exception as exc:  # noqa: BLE001 - network errors are expected and non-fatal
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["title"] = _title_from_url(url)
    return result


def fetch_links(
    urls: List[str],
    max_links: int = 500,
    max_chars: int = 6000,
    timeout: int = 20,
    workers: int = 8,
    progress=None,
) -> List[Dict[str, Optional[str]]]:
    """Fetch up to ``max_links`` URLs concurrently, preserving input order.

    ``progress``, if given, is called as ``progress(index, total, result)`` once
    per link as it completes (index is 1-based in completion order), so callers
    can stream status to stdout.
    """
    selected = urls[:max_links]
    results: Dict[str, Dict[str, Optional[str]]] = {}
    if not selected:
        return []
    total = len(selected)
    done = 0
    with ThreadPoolExecutor(max_workers=min(workers, len(selected))) as pool:
        future_map = {pool.submit(fetch_url, u, max_chars, timeout): u for u in selected}
        for future in as_completed(future_map):
            u = future_map[future]
            try:
                results[u] = future.result()
            except Exception as exc:  # noqa: BLE001
                results[u] = {"url": u, "title": _title_from_url(u), "text": None,
                              "kind": "html", "error": f"{type(exc).__name__}: {exc}",
                              "scholarly": looks_scholarly(u)}
            done += 1
            if progress is not None:
                try:
                    progress(done, total, results[u])
                except Exception:  # noqa: BLE001 - progress must never break fetching
                    pass
    return [results[u] for u in selected if u in results]


def format_link_details(fetched: List[Dict[str, Optional[str]]]) -> str:
    """Render fetched link results into a text block for the LLM prompt."""
    blocks = []
    for item in fetched:
        lines = [f"URL: {item['url']}"]
        if item.get("title"):
            lines.append(f"Title: {item['title']}")
        kind = item.get("kind")
        if kind and kind != "html":
            lines.append(f"Type: {kind}")
        if item.get("error"):
            lines.append(f"Note: {item['error']}")
        if item.get("text"):
            lines.append("Extracted text:")
            lines.append(item["text"])
        blocks.append("\n".join(lines))
    return "\n\n---\n\n".join(blocks)
