"""
Syncs glycerol campaign data to the glycerin-decision GitHub repo.

Run this after each batch to push updated results.
It copies campaign CSVs from the local output folder into the cloned
glycerin-decision repo and pushes to the initial-csv-provisioning branch.

PNG files are excluded to keep the repo size manageable.
"""

import os
import shutil
import subprocess
from datetime import datetime

# ===== CONFIG =====
SOURCE_DIR = r"C:\Users\Imaging Controller\Desktop\utoronto_demo\output\glycerol_sobol_campaign"
DEST_REPO  = r"C:\Users\Imaging Controller\Desktop\glycerin-decision"
DEST_DATA  = os.path.join(DEST_REPO, "data")
BRANCH     = "initial-csv-provisioning"

# File extensions to SKIP (plots - too large for git)
SKIP_EXTENSIONS = {".png"}
# ===================


def _copy_tree(src, dst):
    """Recursively copy src into dst, skipping SKIP_EXTENSIONS. Returns count of files copied."""
    os.makedirs(dst, exist_ok=True)
    count = 0
    for entry in os.scandir(src):
        dest_path = os.path.join(dst, entry.name)
        if entry.is_dir(follow_symlinks=False):
            count += _copy_tree(entry.path, dest_path)
        else:
            if os.path.splitext(entry.name)[1].lower() in SKIP_EXTENSIONS:
                continue
            shutil.copy2(entry.path, dest_path)
            count += 1
    return count


def _git(args, cwd=DEST_REPO):
    result = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout.strip()


def sync():
    print("=== Glycerol Data Sync ===")
    print(f"Source : {SOURCE_DIR}")
    print(f"Dest   : {DEST_DATA}")

    if not os.path.isdir(SOURCE_DIR):
        raise FileNotFoundError(f"Source folder not found: {SOURCE_DIR}")
    if not os.path.isdir(DEST_REPO):
        raise FileNotFoundError(
            f"glycerin-decision repo not found at: {DEST_REPO}\n"
            "Clone it first: git clone https://github.com/AccelerationConsortium/glycerin-decision.git "
            "--branch initial-csv-provisioning Desktop/glycerin-decision"
        )

    # Make sure we're on the right branch
    current_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    if current_branch != BRANCH:
        print(f"Switching from '{current_branch}' to '{BRANCH}'...")
        _git(["checkout", BRANCH])

    # Pull latest before adding files
    print("Pulling latest from remote...")
    _git(["pull", "origin", BRANCH])

    # Copy files
    print("Copying campaign data...")
    copied = _copy_tree(SOURCE_DIR, DEST_DATA)
    print(f"  {copied} files copied (PNGs skipped)")

    # Check if anything changed
    status = _git(["status", "--porcelain"])
    if not status:
        print("Nothing changed — repo already up to date.")
        return

    changed_lines = status.splitlines()
    print(f"  {len(changed_lines)} files changed/added in repo")

    # Stage, commit, push
    _git(["add", "data/"])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    commit_msg = f"sync glycerol campaign data {timestamp}"
    _git(["commit", "-m", commit_msg])
    print(f"Committed: '{commit_msg}'")

    print(f"Pushing to origin/{BRANCH}...")
    _git(["push", "origin", BRANCH])
    print("Done! Data is live on GitHub.")


if __name__ == "__main__":
    sync()
