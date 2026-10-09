"""Harmless child process for testing GUI queue sequencing, never hardware."""
import json
import sys
import time
from pathlib import Path
import yaml

root = Path(sys.argv[1])
job_id = sys.argv[2]
folder = root / "jobs" / job_id
job = json.loads((folder / "input.json").read_text())
if job["config"].get("TEST_OUTPUT_DELAY"):
    print("Child is still running", flush=True)
    print("Child stderr is visible", file=sys.stderr, flush=True)
    time.sleep(job["config"]["TEST_OUTPUT_DELAY"])
with (root / "order.txt").open("a") as stream:
    stream.write(job_id + "\n")
state_path = root / "robot_status.yaml"
if not state_path.exists():
    state_path.write_text(yaml.safe_dump({"pipets_used": {"rack": 0}}))
status = yaml.safe_load(state_path.read_text())
tips_before = status["pipets_used"]["rack"]
status["pipets_used"]["rack"] += 1
(root / "robot_status.yaml").write_text(yaml.safe_dump(status))
failed = job["config"].get("TEST_FAILURE", False)
result = {"completed": not failed, "handoff_ok": not failed,
          "records": [{"level": "WARNING", "message": "test warning"}],
          "tips_used": {"rack": 1}, "tips_before": tips_before}
if failed:
    result["records"].append({"level": "ERROR", "message": "test failure"})
if "--live-fixture" not in sys.argv and job["config"].get("TEST_SIMULATION_ERROR", False):
    result["records"].append({"level": "ERROR", "message": "simulation error with successful exit"})
if "--live-fixture" in sys.argv and job["config"].get("TEST_LIVE_ERROR", False):
    result["records"].append({"level": "ERROR", "message": "live test error with successful exit"})
(folder / "result.json").write_text(json.dumps(result))
print("Harmless simulated child finished")
raise SystemExit(1 if failed else 0)
