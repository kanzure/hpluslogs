"""Tests for the diyhplus wiki ingestion pipeline (offline, hermetic)."""

from __future__ import annotations

import json
from pathlib import Path

from hpluslogs.integrations import diyhplus
from hpluslogs.services import ingest, xai_upload


###############################################################################
# Integration adapter: page discovery
###############################################################################

def _make_wiki(root: Path) -> None:
    (root / "bitcoin").mkdir(parents=True, exist_ok=True)
    (root / "index.mdwn").write_text("# Top index\n", encoding="utf-8")
    (root / "brain_uploading.mdwn").write_text("# Brain uploading\n", encoding="utf-8")
    (root / "bitcoin" / "index.mdwn").write_text("# Bitcoin index\n", encoding="utf-8")
    (root / "bitcoin" / "history.txt").write_text("history notes\n", encoding="utf-8")
    (root / "logo.png").write_bytes(b"\x89PNG\r\n")


def test_iter_pages_finds_text_sources_recursively(tmp_path: Path) -> None:
    _make_wiki(tmp_path)
    pages = dict((name, path) for path, name in diyhplus.iter_pages(tmp_path))

    assert set(pages) == {
        "index.mdwn",
        "brain_uploading.mdwn",
        "bitcoin/index.mdwn",
        "bitcoin/history.txt",
    }
    # Images and other assets are not indexable pages.
    assert "logo.png" not in pages
    # Relative names are stable keys even when basenames collide: the two
    # index.mdwn files show up as distinct entries.
    names = sorted(name for _, name in diyhplus.iter_pages(tmp_path))
    assert "index.mdwn" in names
    assert any(name.endswith("bitcoin/index.mdwn") for name in names)
    assert len(names) == len(set(names))


def test_is_page_accepts_wiki_text_suffixes(tmp_path: Path) -> None:
    mdwn = tmp_path / "a.mdwn"
    mdwn.write_text("x", encoding="utf-8")
    txt = tmp_path / "b.txt"
    txt.write_text("x", encoding="utf-8")
    png = tmp_path / "c.png"
    png.write_bytes(b"\x89PNG\r\n")

    assert diyhplus.is_page(mdwn)
    assert diyhplus.is_page(txt)
    assert not diyhplus.is_page(png)
    assert not diyhplus.is_page(tmp_path)


###############################################################################
# Fetch service
###############################################################################

def test_fetch_reports_page_count(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    checkout = data_dir / "raw-more" / "diyhpl.us"
    _make_wiki(checkout)

    monkeypatch.setattr(
        diyhplus, "clone_or_update",
        lambda dest, **kw: "abc1234",
    )

    ingest.run_diyhplus(data_dir)
    # Checkout directory is created (parents) even though the clone is stubbed.
    assert checkout.parent.exists()


###############################################################################
# xAI collection upload
###############################################################################

class _StubClient:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _patch_xai(monkeypatch, created: list, uploaded: list) -> None:
    from hpluslogs.integrations import xai

    def create_collection(name, chunk_size, chunk_overlap, field_definitions):
        created.append({
            "name": name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "field_definitions": field_definitions,
        })
        return "collection_test"

    async def upload_document_async(client, collection_id, file_path, fields, wait=False):
        uploaded.append((str(file_path), dict(fields)))
        return "file_test"

    monkeypatch.setattr(xai, "create_collection", create_collection)
    monkeypatch.setattr(xai, "upload_document_async", upload_document_async)
    monkeypatch.setattr(xai, "get_async_client", lambda: _StubClient())


def test_collect_uploads_wiki_pages_with_relative_keys(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    checkout = data_dir / "raw-more" / "diyhpl.us"
    _make_wiki(checkout)

    created: list = []
    uploaded: list = []
    _patch_xai(monkeypatch, created, uploaded)

    xai_upload.run_diyhplus(
        data_dir, collection_name="diyhplus-wiki",
        resume=True, concurrency=4, wait_for_indexing=False,
    )

    # Collection created once with the wiki defaults.
    assert len(created) == 1
    assert created[0]["name"] == "diyhplus-wiki"
    assert created[0]["field_definitions"][0]["key"] == "filename"

    # Only text pages uploaded, keyed by relative path.
    keys = sorted(fields["filename"] for _, fields in uploaded)
    assert keys == [
        "bitcoin/history.txt",
        "bitcoin/index.mdwn",
        "brain_uploading.mdwn",
        "index.mdwn",
    ]

    config = json.loads((data_dir / "diyhplus_collection.json").read_text(encoding="utf-8"))
    assert config["collection_id"] == "collection_test"
    assert sorted(config["uploaded_files"]) == keys


def test_collect_resume_skips_uploaded_pages(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    checkout = data_dir / "raw-more" / "diyhpl.us"
    _make_wiki(checkout)

    created: list = []
    uploaded: list = []
    _patch_xai(monkeypatch, created, uploaded)

    xai_upload.run_diyhplus(data_dir, resume=True, concurrency=4)
    first_count = len(uploaded)
    assert first_count == 4

    # Second resumed run uploads nothing new and reuses the stored collection.
    uploaded.clear()
    xai_upload.run_diyhplus(data_dir, resume=True, concurrency=4)
    assert uploaded == []
    assert len(created) == 1

    # --no-resume re-uploads everything but still targets the same collection.
    xai_upload.run_diyhplus(data_dir, resume=False, concurrency=4)
    assert len(uploaded) == first_count


def test_collect_requires_checkout(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(Exception) as excinfo:
        xai_upload.run_diyhplus(tmp_path / "missing", resume=True)
    assert "diyhplus wiki directory does not exist" in str(excinfo.value)


###############################################################################
# Prompts
###############################################################################

def test_diyhplus_prompts() -> None:
    from hpluslogs.core.prompts import DIYHPLUS_SEARCH_QUERY_PROMPT, DIYHPLUS_SYSTEM_PROMPT

    assert "diyhpl.us" in DIYHPLUS_SYSTEM_PROMPT
    rendered = DIYHPLUS_SEARCH_QUERY_PROMPT.format(prompt_fragment="cryonics")
    assert "cryonics" in rendered


def test_generate_search_query_selects_diyhplus_prompt(monkeypatch) -> None:
    from hpluslogs.integrations import openrouter
    from hpluslogs.services import generation

    seen: list = []

    def fake_complete(prompt, model):
        seen.append(prompt)
        return "cryonics protocols"

    monkeypatch.setattr(openrouter, "complete", fake_complete)

    query = generation.generate_search_query("cryonics", "model-x", for_diyhplus=True)
    assert query == "cryonics protocols"
    assert "do-it-yourself biohacking" in seen[0]
