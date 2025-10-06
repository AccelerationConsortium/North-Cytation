"""Minimal fetch implementation (test mode focused).

NOTE: Real HTTP calls to OpenAlex omitted; replace `_mock_openalex_query` with
actual requests.get logic when network usage is enabled.
"""
from __future__ import annotations
import json
import os
from datetime import datetime
from typing import Dict, Iterable, List

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
    return (
        f"ANY: {tokens_any}; ALL: {mandatory_all}; ASSAY: {assay_any}"
    )


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
    for arche, params in arche_cfg.items():
        if arche not in target_archetypes:
            continue
        tokens_any = params.get("tokens_any", [])
        mandatory_all = params.get("mandatory_all", [])
        assay_any = params.get("assay_any", [])

        # Build description (placeholder for actual query string)
        query_descr = build_query_description(tokens_any, mandatory_all, assay_any)

        records = _mock_openalex_query(arche, tokens_any, mandatory_all, assay_any, limit)
        # Filter year & review types (mock data already compliant)
        filtered = [r for r in records if r["publication_year"] >= min_year]
        if exclude_reviews:
            filtered = [r for r in filtered if r.get("type") not in {"review", "Review"}]

        write_jsonl(raw_jsonl, filtered, existing_ids)
        summary.append({
            "archetype": arche,
            "fetched": len(records),
            "written": len(filtered),
            "query": query_descr,
        })

    # Print simple run summary (stdout)
    print("FETCH RUN SUMMARY (TEST MODE=" + str(test_mode) + ")")
    for row in summary:
        print(row)
    print(f"Total unique records now: {len(existing_ids)}")
    print("Timestamp:", datetime.utcnow().isoformat() + "Z")


if __name__ == "__main__":
    main()
