"""Version-aware plate accounting for scheduler state files."""


def source_ids(state):
    if state.get("schema_version") == 2:
        return set(state["positions"].get("source_stack", []))
    return {f"legacy-source_stack-{index + 1:03d}" for index in range(state.get("num_in_source", 0))}


def operation_issues(state):
    pending = state.get("pending_transfer")
    if pending:
        return [f"Plate transfer requires review: {pending}"]
    return []
