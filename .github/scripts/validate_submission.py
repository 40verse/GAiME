#!/usr/bin/env python3
"""
Validate GAiME submission PRs.

Checks:
  1. All changed files live under games/<game>/submissions/
     → Closes the PR if not
  2. Each file is named <handle>_<model>.md
     → Fails the check if not
  3. Each file contains the required metadata fields
     → Fails the check if not
"""

import base64
import json
import os
import re
import subprocess
import sys

REQUIRED_FIELDS = [
    "**Handle:**",
    "**Model:**",
    "**Method:**",
    "**Link:**",
    "**Usage:**",
]

# games/<anything>/submissions/<file>.md  (no subdirectories in submissions/)
VALID_PATH_RE = re.compile(r"^games/[^/]+/submissions/[^/]+\.md$")

# <handle>_<model>.md  — no spaces, underscore as separator
VALID_FILENAME_RE = re.compile(r"^[^\s_]+_[^\s]+\.md$")

PR_NUMBER = os.environ["PR_NUMBER"]
PR_SHA = os.environ["PR_SHA"]
REPO = os.environ["REPO"]


def gh(*args):
    subprocess.run(["gh"] + list(args), check=True)


def gh_api(path):
    result = subprocess.run(
        ["gh", "api", path],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def get_pr_files():
    data = gh_api(f"repos/{REPO}/pulls/{PR_NUMBER}/files")
    return [
        f for f in data
        if f["status"] in ("added", "modified", "renamed")
        and not f["filename"].endswith(".gitkeep")
    ]


def get_file_content(path):
    data = gh_api(f"repos/{REPO}/contents/{path}?ref={PR_SHA}")
    return base64.b64decode(data["content"]).decode("utf-8")


def comment(body):
    gh("pr", "comment", PR_NUMBER, "--body", body, "--repo", REPO)


def main():
    files = get_pr_files()

    if not files:
        print("No relevant files changed — nothing to validate.")
        sys.exit(0)

    # ── Check 1: all files must be inside games/*/submissions/ ──────────────
    outside = [f["filename"] for f in files if not VALID_PATH_RE.match(f["filename"])]
    if outside:
        comment(
            "## Submission Rejected\n\n"
            "This PR touches files outside the allowed `games/*/submissions/` path:\n\n"
            + "\n".join(f"- `{p}`" for p in outside)
            + "\n\nOnly `games/<game>/submissions/<handle>_<model>.md` files are accepted. "
            "Please read the [submission guide](../../README.md) and open a new PR."
        )
        gh("pr", "close", PR_NUMBER, "--repo", REPO)
        sys.exit(1)

    # ── Checks 2 & 3: filename pattern + required metadata fields ────────────
    errors = []
    for f in files:
        path = f["filename"]
        filename = os.path.basename(path)

        if not VALID_FILENAME_RE.match(filename):
            errors.append(
                f"**`{path}`** — filename must be `<handle>_<model>.md` "
                f"(e.g. `alice_opus-4-6.md`). Got: `{filename}`"
            )
            continue

        try:
            content = get_file_content(path)
        except Exception as e:
            errors.append(f"**`{path}`** — could not read file: {e}")
            continue

        missing = [field for field in REQUIRED_FIELDS if field not in content]
        if missing:
            errors.append(
                f"**`{path}`** — missing required metadata: "
                + ", ".join(f"`{m}`" for m in missing)
            )

    if errors:
        comment(
            "## Submission Needs Fixes\n\n"
            + "\n".join(f"- {e}" for e in errors)
            + "\n\nSee the [submission template](README.md#submission-template) "
            "for the correct format. Push a fix to this branch — no need to open a new PR."
        )
        sys.exit(1)

    print("Validation passed.")


if __name__ == "__main__":
    main()
