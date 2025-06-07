from __future__ import annotations
import os
import time
import difflib
from github import Github


def create_hitl_pr(
    patch: dict,
    summary: str,
    repo_name: str | None = None,
) -> str:
    """Create a pull request with the given patch and validation summary.

    Parameters
    ----------
    patch : dict
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
    repo = gh.get_repo(repo_name)

    base_branch = repo.default_branch
    base = repo.get_branch(base_branch)
    branch_name = f"hitl-{int(time.time())}"
    repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base.commit.sha)

    file_path = patch["target"]
    after = patch.get("after", "")

    try:
        contents = repo.get_contents(file_path, ref=base_branch)
        repo.update_file(
            file_path,
            f"Apply patch via HITL workflow",
            after,
            contents.sha,
            branch=branch_name,
        )
    except Exception:
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

    pr = repo.create_pull(
        title=f"HITL Patch {branch_name}",
        body=body,
        head=branch_name,
        base=base_branch,
    )
    return pr.html_url
