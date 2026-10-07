"""Harmless child process for testing GUI queue sequencing, never hardware."""
import json
import sys
from pathlib import Path
import yaml

root = Path(sys.argv[1])
job_id = sys.argv[2]
folder = root / "jobs" / job_id
job = json.loads((folder / "input.json").read_text())
with (root / "order.txt").open("a") as stream:
    stream.write(job_id + "\n")
status = yaml.safe_load((root / "robot_status.yaml").read_text())
tips_before = status["pipets_used"]["rack"]
status["pipets_used"]["rack"] += 1
(root / "robot_status.yaml").write_text(yaml.safe_dump(status))
failed = job["config"].get("TEST_FAILURE", False)
result = {"completed": not failed, "handoff_ok": not failed,
          "records": [{"level": "WARNING", "message": "test warning"}],
          "tips_used": {"rack": 1}, "tips_before": tips_before}
if failed:
    result["records"].append({"level": "ERROR", "message": "test failure"})
(folder / "result.json").write_text(json.dumps(result))
print("Harmless simulated child finished")
raise SystemExit(1 if failed else 0)