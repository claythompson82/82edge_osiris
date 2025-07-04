from __future__ import annotations

import argparse
import difflib
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

from github import Github
from github.PullRequest import PullRequest

if TYPE_CHECKING:
    from github.Repository import Repository


class PatchDict(TypedDict):
    """Structure for a patch operation."""

    target: str
    before: str
    after: str


class TemplateVars(TypedDict, total=False):
    pr_number: int
    summary: str
    score: float


TEMPLATE_VARS: TemplateVars = {}


def create_hitl_pr(
    patch: PatchDict,
    summary: str,
    repo_name: str | None = None,
) -> str:
    """Create a pull request with the given patch and validation summary.

    Parameters
    ----------
    patch : PatchDict
        Dictionary with keys ``target``, ``before`` and ``after``.
    summary : str
        Summary text from the verification process.
    repo_name : str, optional
        GitHub repository in the form ``owner/repo``. If omitted,
        ``GITHUB_REPOSITORY`` environment variable will be used.

    Returns
    -------
    str
        URL of the created pull request.
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN not set")

    repo_name = repo_name or os.environ.get("GITHUB_REPOSITORY")
    if not repo_name:
        raise RuntimeError("Repository name not provided")

    gh = Github(token)
    repo: Repository = gh.get_repo(repo_name)

    base_branch_name = repo.default_branch
    base = repo.get_branch(base_branch_name)
    branch_name = f"hitl-{int(time.time())}"
    repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base.commit.sha)

    file_path = patch["target"]
    after = patch.get("after", "")

    try:
        contents = cast(Any, repo.get_contents(file_path, ref=base_branch_name))
        repo.update_file(
            file_path,
            "Apply patch via HITL workflow",
            after,
            contents.sha,
            branch=branch_name,
        )
    except Exception:  # Should be more specific if possible, e.g., UnknownObjectException
        repo.create_file(
            file_path,
            f"Add {file_path}",
            after,
            branch=branch_name,
        )

    before = patch.get("before", "")
    diff = "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile="before",
            tofile="after",
        )
    )

    body = (
        "### Patch Diff\n\n" f"```diff\n{diff}\n```\n" + "\n### Validation Summary\n" + summary
    )

    pr: PullRequest = repo.create_pull(
        title=f"HITL Patch {branch_name}",
        body=body,
        head=branch_name,
        base=base_branch_name,
    )
    return str(pr.html_url)


def render_pr_comment(template_path: str, template_vars: TemplateVars) -> str:
    """Return a formatted PR comment from a template file."""
    template = Path(template_path).read_text()
    comment = template.format(**template_vars)
    return comment


def comment_on_pr(pr_number: int, body: str, repo_name: str | None = None) -> None:
    """Post a comment to a pull request.

    Parameters
    ----------
    pr_number : int
        Pull request number.
    body : str
        Comment body.
    repo_name : str, optional
        GitHub repository in the form ``owner/repo``. If omitted,
        ``GITHUB_REPOSITORY`` environment variable will be used.
    """
    token = os.environ.get("GITHUB_TOKEN")
    repo_name = repo_name or os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo_name:
        raise RuntimeError("GitHub environment not configured")

    gh = Github(token)
    repo: Repository = gh.get_repo(repo_name)
    pr: PullRequest = repo.get_pull(pr_number)
    pr.create_issue_comment(body)


def cli(argv: list[str] | None = None) -> None:
    """CLI entry-point for posting a HITL comment."""
    parser = argparse.ArgumentParser(description="Post a comment to a PR")
    parser.add_argument("--pr", type=int, required=True, help="pull request number")
    parser.add_argument("--msg", required=True, help="comment body")
    args = parser.parse_args(argv)
    comment_on_pr(args.pr, args.msg)


if __name__ == "__main__":  # pragma: no cover - manual CLI
    cli()
