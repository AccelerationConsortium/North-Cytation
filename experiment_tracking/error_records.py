"""Structured error records: turn a run's exception into a short, searchable record.

Only adds information. Nothing here raises: any problem reading the codes list or the
exception falls back to code UNKNOWN, so a run never fails or changes because of it.
Codes and their kind / what-to-do text live in error_codes.yaml next to this file.
"""
import os
import traceback

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_MODULE_DIR)
CODES_FILE = os.path.join(_MODULE_DIR, "error_codes.yaml")

UNKNOWN = {"code": "UNKNOWN", "kind": "unknown", "device": "",
           "what_to_do": "Not a known error. See the traceback.", "slack": True}


def load_codes(path=CODES_FILE):
    try:
        import yaml
        with open(path, encoding="utf-8") as stream:
            return (yaml.safe_load(stream) or {}).get("codes") or {}
    except Exception:
        return {}


def _chain(error):
    """The error, then whatever it wraps (raise ... from e, or raised while handling e)."""
    seen = []
    while error is not None and all(error is not item for item in seen) and len(seen) < 10:
        seen.append(error)
        error = error.__cause__ or error.__context__
    return seen


def _c9_error(error):
    """(source, number) from the North SDK's RuntimeError(C9Errors(...)), else None."""
    detail = error.args[0] if getattr(error, "args", None) else None
    if hasattr(detail, "source") and hasattr(detail, "error"):
        return detail.source, detail.error
    return None


def _as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _match(error, codes):
    chain = _chain(error)
    for item in chain:
        code = getattr(item, "error_code", None)
        if code in codes:
            return code, item
    for item in chain:
        c9 = _c9_error(item)
        if c9 is not None:
            for code, entry in codes.items():
                if c9[1] in _as_list(entry.get("c9_error")):
                    return code, item
    for item in chain:
        text = str(item)
        for code, entry in codes.items():
            if any(part in text for part in _as_list(entry.get("message_contains"))):
                return code, item
    for item in chain:
        for code, entry in codes.items():
            if type(item).__name__ in _as_list(entry.get("exception")):
                return code, item
    return None, error


def _step(error):
    """Deepest frame in our own repo code: where the error was noticed."""
    step = ""
    for frame in traceback.extract_tb(error.__traceback__):
        path = os.path.abspath(frame.filename)
        # Skip the hardware packages and the simulated controller: the step is in our code
        if (path.startswith(_REPO_ROOT) and "site-packages" not in path
                and os.path.basename(path) != "fault_injection.py"):
            step = f"{frame.name} ({os.path.relpath(path, _REPO_ROOT)}:{frame.lineno})"
    return step


def classify(error, codes=None):
    """A dict describing the error. Never raises."""
    try:
        codes = load_codes() if codes is None else codes
        code, matched = _match(error, codes)
        entry = dict(UNKNOWN) if code is None else {**UNKNOWN, **codes[code], "code": code}
        c9 = _c9_error(matched)
        return {
            "code": entry["code"],
            "kind": entry.get("kind", "unknown"),
            "device": entry.get("device", ""),
            "device_code": f"{c9[0]}/{c9[1]}" if c9 is not None else "",
            "exception": type(matched).__name__,
            "message": str(matched),
            "step": _step(error),
            "what_to_do": entry.get("what_to_do", ""),
            "slack": bool(entry.get("slack", True)),
        }
    except Exception as problem:
        return {**UNKNOWN, "device_code": "", "exception": type(error).__name__,
                "message": f"{error} (could not classify: {problem})", "step": ""}


def write_record(path, record):
    """Write one record as YAML. Returns True if written. Never raises."""
    try:
        import yaml
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as stream:
            yaml.safe_dump(record, stream, sort_keys=False, allow_unicode=True)
        return True
    except Exception:
        return False
