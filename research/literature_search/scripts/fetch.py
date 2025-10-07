"""Fetch stage.

Supports two modes:
 - Test mode (mock records, deterministic, no network)
 - Real mode (OpenAlex API) with minimal pagination & filtering

Design principles: small, side-effect free functions; schema-consistent
records whether mock or real; minimal coupling to archetype semantics.
"""
from __future__ import annotations
import json
import os
from datetime import datetime
from typing import Dict, Iterable, List, Optional
import time
import urllib.parse
import urllib.request
import ssl

import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _mock_openalex_query(archetype: str, tokens_any: List[str], mandatory_all: List[str], assay_any: List[str], limit: int) -> List[Dict]:
    """Return synthetic OpenAlex-like records for offline test mode.

    Structure mimics a subset of the real /works endpoint fields we care about.
    """
    results = []
    for i in range(limit):
        abstract_text = (
            f"This study reports {tokens_any[0]} applied to polymer systems with "
            f"kinetic sampling and fluorescence measurement optimizing catalyst loading."
        )
        results.append(
            {
                "id": f"W_TEST_{archetype}_{i}",
                "doi": f"10.9999/test.{archetype}.{i}",
                "title": f"Test {archetype.replace('_',' ')} paper {i}",
                "publication_year": 2023,
                "type": "journal-article",
                "cited_by_count": 12 + i,
                "abstract": abstract_text,
                "concepts": ["polymer", archetype],
            }
        )
    return results


def build_query_description(tokens_any: List[str], mandatory_all: List[str], assay_any: List[str]) -> str:
    return f"ANY: {tokens_any}; ALL: {mandatory_all}; ASSAY: {assay_any}"


def _compose_search_string(tokens_any: List[str], mandatory_all: List[str], assay_any: List[str]) -> str:
    """Return a conservative search string acceptable to OpenAlex.

    OpenAlex `search=` is a plain text query (no advanced boolean guaranteed).
    We therefore just concatenate unique tokens (spaces) to avoid 400 errors
    caused by parentheses / unsupported syntax.
    """
    seen = set()
    ordered: List[str] = []
    for group in (tokens_any, mandatory_all, assay_any):
        for tok in group:
            t = tok.strip()
            if not t:
                continue
            if t not in seen:
                seen.add(t)
                ordered.append(t)
    return " ".join(ordered)


def _reconstruct_abstract(inv_idx: dict) -> str:
    # OpenAlex returns abstract_inverted_index: {word: [positions]}
    if not inv_idx:
        return ""
    # Build position -> word mapping
    flat = []
    for word, positions in inv_idx.items():
        for pos in positions:
            flat.append((pos, word))
    if not flat:
        return ""
    flat.sort(key=lambda x: x[0])
    max_pos = flat[-1][0]
    words = [""] * (max_pos + 1)
    for pos, word in flat:
        words[pos] = word
    return " ".join(w for w in words if w)


def _openalex_request(url: str, timeout: float = 10.0) -> Optional[dict]:
    try:
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(url, timeout=timeout, context=ctx) as resp:
            import json as _json
            return _json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"WARN: OpenAlex request failed: {e} -> {url}")
        return None


def _fetch_openalex(archetype: str, tokens_any: List[str], mandatory_all: List[str], assay_any: List[str], min_year: int, limit: int, exclude_reviews: bool) -> List[Dict]:
    # Compose search (OpenAlex full-text search uses `search=` param across title+abstract)
    query_text = _compose_search_string(tokens_any, mandatory_all, assay_any)
    encoded = urllib.parse.quote(query_text)
    per_page = min(limit, 25)
    remaining = limit
    cursor = "*"
    results: List[Dict] = []
    base = "https://api.openalex.org/works"
    while remaining > 0 and cursor:
        page_url = (
            f"{base}?search={encoded}&filter=from_publication_date:{min_year}-01-01"
            f"&per-page={per_page}&cursor={urllib.parse.quote(cursor)}&mailto=research@placeholder.org"
        )
        data = _openalex_request(page_url)
        if not data:
            break
        works = data.get("results", [])
        for w in works:
            if exclude_reviews and w.get("type") in {"review"}:
                continue
            abstract = w.get("abstract") or ""
            if not abstract and "abstract_inverted_index" in w:
                abstract = _reconstruct_abstract(w.get("abstract_inverted_index") or {})
            rec = {
                "id": w.get("id"),
                "doi": w.get("doi"),
                "title": w.get("title"),
                "publication_year": w.get("publication_year"),
                "type": w.get("type"),
                "cited_by_count": w.get("cited_by_count", 0),
                "abstract": abstract,
                "concepts": [c.get("display_name") for c in (w.get("concepts") or []) if c.get("display_name")],
                "archetype": archetype,
            }
            results.append(rec)
            if len(results) >= limit:
                break
        cursor = data.get("meta", {}).get("next_cursor")
        remaining = limit - len(results)
        if remaining > 0:
            time.sleep(0.8)  # gentle rate limiting
        if not cursor:
            break
    return results


def _fetch_openalex_iterative_union(archetype: str, tokens_all: List[str], min_year: int, limit: int, exclude_reviews: bool, pages: int = 1, per_page: int = 5) -> List[Dict]:
    """Fallback strategy: query each token separately and union results until limit.

    Rationale: OpenAlex `search` uses AND semantics across space-separated terms; long broad queries may return zero.
    This pulls a small slice per token and merges unique works.
    """
    seen_ids = set()
    union: List[Dict] = []
    base = "https://api.openalex.org/works"
    per_token_page = max(1, min(per_page, 50))  # OpenAlex typical practical upper bound
    for tok in tokens_all:
        if len(union) >= limit:
            break
        cursor = "*"
        fetched_pages = 0
        while cursor and fetched_pages < pages and len(union) < limit:
            qt = urllib.parse.quote(tok)
            page_url = (
                f"{base}?search={qt}&filter=from_publication_date:{min_year}-01-01"
                f"&per-page={per_token_page}&cursor={urllib.parse.quote(cursor)}"
                f"&mailto=research@placeholder.org"
            )
            data = _openalex_request(page_url)
            if not data:
                break
            for w in data.get("results", []):
                wid = w.get("id")
                if not wid or wid in seen_ids:
                    continue
                if exclude_reviews and w.get("type") in {"review"}:
                    continue
                abstract = w.get("abstract") or ""
                if not abstract and "abstract_inverted_index" in w:
                    abstract = _reconstruct_abstract(w.get("abstract_inverted_index") or {})
                rec = {
                    "id": wid,
                    "doi": w.get("doi"),
                    "title": w.get("title"),
                    "publication_year": w.get("publication_year"),
                    "type": w.get("type"),
                    "cited_by_count": w.get("cited_by_count", 0),
                    "abstract": abstract,
                    "concepts": [c.get("display_name") for c in (w.get("concepts") or []) if c.get("display_name")],
                    "archetype": archetype,
                    "query_token": tok,
                }
                union.append(rec)
                seen_ids.add(wid)
                if len(union) >= limit:
                    break
            cursor = data.get("meta", {}).get("next_cursor")
            fetched_pages += 1
            time.sleep(0.4)
            if len(union) >= limit:
                break
    return union


def write_jsonl(path: str, records: Iterable[Dict], existing_ids: set):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            if r["id"] in existing_ids:
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            existing_ids.add(r["id"])


def load_existing_ids(path: str) -> set:
    ids = set()
    if not os.path.exists(path):
        return ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                ids.add(obj.get("id"))
            except Exception:
                continue
    return ids


def main():
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "literature_pipeline_config.yaml"
    )
    config_path = os.path.abspath(config_path)
    cfg = load_config(config_path)

    fetch_cfg = cfg["fetch"]
    arche_cfg = cfg["archetypes"]

    test_mode = fetch_cfg.get("test_mode", False)
    if test_mode:
        target_archetypes = fetch_cfg.get("test_archetypes", []) or list(arche_cfg.keys())[:1]
        limit = fetch_cfg.get("test_limit_per_archetype", 5)
    else:
        target_archetypes = list(arche_cfg.keys())
        limit = fetch_cfg.get("max_per_archetype", 500)

    raw_jsonl = fetch_cfg["raw_jsonl"]
    existing_ids = load_existing_ids(raw_jsonl)

    min_year = fetch_cfg.get("min_year", 2015)
    exclude_reviews = fetch_cfg.get("exclude_reviews", True)

    summary = []
    # For initial real run keep it small: only first archetype if test_mode False and limit very small
    processed = 0
    run_tag = fetch_cfg.get("run_tag")
    for arche, params in arche_cfg.items():
        if arche not in target_archetypes:
            continue
        if not test_mode and processed >= 1:
            break  # limit to first archetype for quick validation

        tokens_any = params.get("tokens_any", [])
        union_tokens = params.get("union_tokens", [])
        all_union_pool = list(dict.fromkeys(list(tokens_any) + list(union_tokens)))  # preserve order
        mandatory_all = params.get("mandatory_all", [])
        assay_any = params.get("assay_any", [])

        # Build description (placeholder for actual query string)
        query_descr = build_query_description(tokens_any, mandatory_all, assay_any)

        if test_mode:
            records = _mock_openalex_query(arche, tokens_any, mandatory_all, assay_any, limit)
        else:
            records = _fetch_openalex(arche, tokens_any, mandatory_all, assay_any, min_year, limit, exclude_reviews)
            # Fallback: if zero fetched (possible 400 previously) retry with tokens_any only
            if not records and tokens_any:
                print(f"Fallback: retrying archetype {arche} with tokens_any only")
                records = _fetch_openalex(arche, tokens_any, [], [], min_year, limit, exclude_reviews)
            if not records and tokens_any:
                print(f"Iterative union strategy for archetype {arche} (core + union tokens: {len(all_union_pool)})")
                iu_pages = fetch_cfg.get("iterative_union_pages", 1)
                iu_per_page = fetch_cfg.get("iterative_union_per_page", 5)
                records = _fetch_openalex_iterative_union(
                    arche, all_union_pool, min_year, limit, exclude_reviews,
                    pages=iu_pages, per_page=iu_per_page
                )
        # Filter year & review types (mock data already compliant)
        filtered = [r for r in records if r["publication_year"] >= min_year]
        if exclude_reviews:
            filtered = [r for r in filtered if r.get("type") not in {"review", "Review"}]

        # Attach run_tag if present
        if run_tag:
            for fr in filtered:
                fr["run_tag"] = run_tag
        write_jsonl(raw_jsonl, filtered, existing_ids)
        summary.append({
            "archetype": arche,
            "fetched": len(records),
            "written": len(filtered),
            "query": query_descr,
        })
        processed += 1

    # Print simple run summary (stdout)
    print("FETCH RUN SUMMARY (TEST MODE=" + str(test_mode) + ")")
    for row in summary:
        print(row)
    print(f"Total unique records now: {len(existing_ids)}")
    print("Timestamp:", datetime.utcnow().isoformat() + "Z")


if __name__ == "__main__":
    main()
