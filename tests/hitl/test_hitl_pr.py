import os
from unittest import mock

from dgm_kernel.hitl_pr import create_hitl_pr


@mock.patch("dgm_kernel.hitl_pr.Github")
def test_create_hitl_pr_makes_pull_request(MockGithub):
    repo = MockGithub.return_value.get_repo.return_value
    repo.default_branch = "main"
    branch = mock.Mock()
    branch.commit.sha = "abc123"
    repo.get_branch.return_value = branch
    repo.get_contents.return_value.sha = "def456"

    patch = {"target": "file.py", "before": "a\n", "after": "a\nb\n"}
    summary = "All checks passed"

    pr = mock.Mock(html_url="http://example.com/pr/1")
    repo.create_pull.return_value = pr

    os.environ["GITHUB_TOKEN"] = "x"
    url = create_hitl_pr(patch, summary, repo_name="owner/repo")

    repo.create_git_ref.assert_called_once()
    repo.update_file.assert_called_once()
    repo.create_pull.assert_called_once()
    assert url == pr.html_url
    body = repo.create_pull.call_args.kwargs["body"]
    assert "```diff" in body
    assert summary in body
