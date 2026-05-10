"""Catalog loader for SHL assessment recommendation system.

Provides a Pydantic `CatalogItem` model and functions to load and
normalize a catalog JSON file into a list of `CatalogItem` objects.

Functions:
- build_searchable_text(item: dict) -> str
- load_catalog(path: str = "data/shl_catalog.json") -> List[CatalogItem]

The loader is defensive: it handles missing fields, normalizes lists,
strips whitespace, skips invalid entries, and logs useful information.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError, validator

logger = logging.getLogger(__name__)


def _normalize_to_list(value: Any) -> List[str]:
	"""Normalize various list-like inputs into a clean list of strings.

	- If value is a list/tuple, each element is coerced to str and stripped.
	- If value is a string, it is split on commas/semicolons/pipes and
	  whitespace, then stripped.
	- None/empty input returns an empty list.
	Duplicate and empty entries are removed while preserving order.
	"""
	if value is None:
		return []
	items: List[str] = []
	if isinstance(value, (list, tuple)):
		iterable = value
	else:
		# Treat scalars as comma/semicolon/pipe-separated values
		iterable = re.split(r"[,;|]+", str(value))
	seen = set()
	for part in iterable:
		if part is None:
			continue
		s = str(part).strip()
		if not s:
			continue
		key = s.lower()
		if key in seen:
			continue
		seen.add(key)
		items.append(s)
	return items


def _parse_bool(value: Any) -> bool:
	"""Interpret common boolean-like values robustly."""
	if isinstance(value, bool):
		return value
	if value is None:
		return False
	text = str(value).strip().lower()
	return text in {"1", "true", "t", "yes", "y", "on"}


def build_searchable_text(item: Dict[str, Any]) -> str:
	"""Combine relevant fields into one clean, lowercase searchable string.

	The following fields are concatenated (when present):
	- name
	- description
	- keys (if present)
	- job_levels
	- languages
	- duration

	The resulting string is lower-cased, punctuation-normalized, and
	whitespace-collapsed.
	"""
	parts: List[str] = []
	name = (item.get("name") or "")
	description = (item.get("description") or "")
	keys = _normalize_to_list(item.get("keys") or item.get("tags"))
	job_levels = _normalize_to_list(item.get("job_levels") or item.get("levels"))
	languages = _normalize_to_list(item.get("languages"))
	duration = str(item.get("duration") or "").strip()

	if name:
		parts.append(str(name))
	if description:
		parts.append(str(description))
	if keys:
		parts.append(" ".join(keys))
	if job_levels:
		parts.append(" ".join(job_levels))
	if languages:
		parts.append(" ".join(languages))
	if duration:
		parts.append(duration)

	raw = " ".join(parts)
	# Normalize: lowercase, replace non-word characters with spaces, collapse spaces
	text = raw.lower()
	text = re.sub(r"[^\w\s]", " ", text)
	text = re.sub(r"\s+", " ", text).strip()
	return text


class CatalogItem(BaseModel):
	"""Pydantic model representing a single catalog item."""

	entity_id: str = Field(..., description="Unique entity identifier")
	name: str = Field(..., description="Human-readable name")
	url: Optional[str] = Field(None, description="URL to the assessment or details")
	description: Optional[str] = Field(None, description="Longer description text")
	job_levels: List[str] = Field(default_factory=list)
	languages: List[str] = Field(default_factory=list)
	duration: Optional[str] = Field(None, description="Duration or estimated time")
	remote: bool = Field(False, description="Is the assessment remote-enabled")
	adaptive: bool = Field(False, description="Is the assessment adaptive")
	test_types: List[str] = Field(default_factory=list)
	keys: List[str] = Field(default_factory=list, description="Catalog categories/keys (e.g., 'Knowledge & Skills', 'Personality & Behavior')")
	searchable_text: str = Field("", description="Precomputed searchable text")

	# Validators to ensure normalized lists
	@validator("job_levels", "languages", "test_types", "keys", pre=True, each_item=False)
	def _validate_lists(cls, v):
		return _normalize_to_list(v)

	@validator("name", "entity_id", pre=True, always=True)
	def _strip_strings(cls, v):
		return str(v).strip() if v is not None else v


def load_catalog(path: str = "data/shl_catalog.json") -> List[CatalogItem]:
	"""Load a JSON catalog and return a list of validated `CatalogItem` objects.

	The loader handles multiple possible shapes (list of items, or a dict
	containing an `items` list). It is defensive: entries missing required
	fields (`entity_id` or `name`) are skipped and logged.
	"""
	p = Path(path)
	if not p.exists():
		logger.error("Catalog file not found: %s", p)
		return []

	try:
		with p.open("r", encoding="utf-8") as fh:
			data = json.load(fh)
	except Exception as exc:  # pragma: no cover - IO/JSON errors
		logger.exception("Failed to read/parse catalog file %s: %s", p, exc)
		return []

	# Support either a top-level list or a dict with an 'items' key
	if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
		entries = data["items"]
	elif isinstance(data, list):
		entries = data
	else:
		logger.error("Unexpected catalog format in %s", p)
		return []

	items: List[CatalogItem] = []
	skipped = 0
	for idx, raw in enumerate(entries):
		if not isinstance(raw, dict):
			logger.warning("Skipping non-object entry at index %d", idx)
			skipped += 1
			continue

		# Required minimal fields
		entity_id = (raw.get("entity_id") or raw.get("id") or "").strip()
		name = (raw.get("name") or "").strip()
		if not entity_id or not name:
			logger.warning(
				"Skipping entry missing required fields at index %d: entity_id=%r, name=%r",
				idx,
				entity_id,
				name,
			)
			skipped += 1
			continue

		# Prepare normalized payload for the model
		payload: Dict[str, Any] = {}
		payload["entity_id"] = entity_id
		payload["name"] = name
		url = raw.get("url") or raw.get("link")
		payload["url"] = str(url).strip() if url is not None else None
		payload["description"] = (raw.get("description") or "").strip() or None
		payload["job_levels"] = raw.get("job_levels") or raw.get("levels") or []
		payload["languages"] = raw.get("languages") or raw.get("language") or []
		payload["duration"] = str(raw.get("duration") or "").strip() or None
		payload["remote"] = _parse_bool(raw.get("remote"))
		payload["adaptive"] = _parse_bool(raw.get("adaptive"))
		payload["test_types"] = raw.get("test_types") or raw.get("types") or []
		payload["keys"] = raw.get("keys") or raw.get("tags") or []

		# Precompute searchable text for faster search indexing
		try:
			payload["searchable_text"] = build_searchable_text({**raw, **payload})
		except Exception:
			payload["searchable_text"] = ""

		try:
			item = CatalogItem(**payload)
			items.append(item)
		except ValidationError as exc:  # pragma: no cover - validation safety
			logger.warning(
				"Skipping invalid catalog entry at index %d (entity_id=%s): %s",
				idx,
				entity_id,
				exc,
			)
			skipped += 1
			continue

	logger.info(
		"Loaded catalog from %s: %d entries processed, %d valid, %d skipped",
		p,
		len(entries),
		len(items),
		skipped,
	)
	return items


__all__ = ["CatalogItem", "load_catalog", "build_searchable_text"]

