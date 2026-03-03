#!/usr/bin/env python3
"""
Update SUBMISSIONS.md registries when a submission PR is merged.

Reads metadata from newly merged submission files and appends a row
to the corresponding game's SUBMISSIONS.md.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

PR_NUMBER = os.environ["PR_NUMBER"]
PR_SHA = os.environ["PR_SHA"]   # merge commit SHA
REPO = os.environ["REPO"]

VALID_PATH_RE = re.compile(r"^games/([^/]+)/submissions/[^/]+\.md$")

# Matches:  **Handle:** some value
FIELD_RE = re.compile(r"^\*\*(\w[\w\s]*):\*\*\s*(.+)", re.MULTILINE)


def gh_api(path):
    result = subprocess.run(
        ["gh", "api", path],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def get_pr_files():
    data = gh_api(f"repos/{REPO}/pulls/{PR_NUMBER}/files")
    return [
        f["filename"] for f in data
        if f["status"] in ("added", "modified")
        and not f["filename"].endswith(".gitkeep")
        and VALID_PATH_RE.match(f["filename"])
    ]


def parse_metadata(content):
    return {m.group(1).strip(): m.group(2).strip() for m in FIELD_RE.finditer(content)}


def ensure_registry(registry_path, game_name):
    if not registry_path.exists():
        title = game_name.replace("-", " ").title()
        registry_path.write_text(
            f"# Submissions — {title}\n\n"
            "Auto-updated when a submission PR is merged. Each row is a verified run.\n\n"
            "| Handle | Model | Method | Link | Usage |\n"
            "|--------|-------|--------|------|-------|\n"
        )


def append_row(registry_path, fields, pr_number):
    handle = fields.get("Handle", "unknown")
    model  = fields.get("Model",  "unknown")
    method = fields.get("Method", "unknown")
    usage  = fields.get("Usage",  "unknown")

    raw_link = fields.get("Link", "")
    link = f"[link]({raw_link})" if raw_link.startswith("http") else raw_link or "—"

    row = f"| {handle} | {model} | {method} | {link} | {usage} |\n"
    registry_path.write_text(registry_path.read_text() + row)
    print(f"Appended to {registry_path}: {row.strip()}")


def main():
    paths = get_pr_files()
    if not paths:
        print("No submission files found in this PR.")
        sys.exit(0)

    for path in paths:
        game = VALID_PATH_RE.match(path).group(1)
        content = Path(path).read_text()
        fields = parse_metadata(content)

        registry_path = Path(f"games/{game}/SUBMISSIONS.md")
        ensure_registry(registry_path, game)
        append_row(registry_path, fields, PR_NUMBER)


if __name__ == "__main__":
    main()
