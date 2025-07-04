from __future__ import annotations

import json
import logging
import os
import sys

log = logging.getLogger(__name__)


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:  # pragma: no cover - stdin issues
        log.error("invalid JSON input: %s", exc)
        return

    token = os.environ.get("GITHUB_TOKEN")
    repo_name = os.environ.get("GITHUB_REPOSITORY")
    pr_number = os.environ.get("PR_NUMBER")

    if not token or not repo_name or not pr_number:
        log.info("GitHub environment not fully configured; skipping comment")
        return

    try:
        from github import Github  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        log.error("PyGithub not available: %s", exc)
        return

    try:
        gh = Github(token)
        repo = gh.get_repo(repo_name)
        pr = repo.get_pull(int(pr_number))
        pr.create_issue_comment(
            f"Patch {payload.get('patch_id')} reward: {payload.get('reward')}"
        )
    except Exception as exc:  # pragma: no cover - network issues
        log.error("failed to post comment: %s", exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
