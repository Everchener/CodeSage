from __future__ import annotations

import requests

from codesage.core.config import GITHUB_HTTP_TIMEOUT_SECONDS, GITHUB_TOKEN

BASE = "https://api.github.com"
USER_AGENT = "codesage/pr-review"


class GitHubTransportError(RuntimeError):
    """在 GitHub HTTP 请求失败时抛出。"""


def _build_headers(accept: str) -> dict[str, str]:
    headers = {
        "Accept": accept,
        "User-Agent": USER_AGENT,
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    return headers


def _format_error_reason(exc: requests.RequestException) -> str:
    if isinstance(exc, requests.Timeout):
        return "timed out"

    if isinstance(exc, requests.HTTPError):
        response = exc.response
        if response is not None:
            status_code = getattr(response, "status_code", "")
            reason = getattr(response, "reason", "") or ""
            status_text = f"{status_code} {reason}".strip()
            if status_text:
                return status_text

    message = str(exc).strip()
    return message or exc.__class__.__name__


def _raise_transport_error(
    operation: str,
    repo: str,
    pr_number: int,
    exc: requests.RequestException,
) -> None:
    reason = _format_error_reason(exc)
    raise GitHubTransportError(
        f"failed to {operation} for {repo}#{pr_number}: {reason}"
    ) from exc


def get_pr_diff(repo: str, pr_number: int) -> str:
    url = f"{BASE}/repos/{repo}/pulls/{pr_number}"
    try:
        response = requests.get(
            url,
            headers=_build_headers("application/vnd.github.v3.diff"),
            timeout=GITHUB_HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        _raise_transport_error("fetch PR diff", repo, pr_number, exc)
    except requests.ConnectionError as exc:
        _raise_transport_error("fetch PR diff", repo, pr_number, exc)
    except requests.HTTPError as exc:
        _raise_transport_error("fetch PR diff", repo, pr_number, exc)
    except requests.RequestException as exc:
        _raise_transport_error("fetch PR diff", repo, pr_number, exc)
    return response.text


def post_review_comment(repo: str, pr_number: int, body: str) -> dict:
    url = f"{BASE}/repos/{repo}/issues/{pr_number}/comments"
    try:
        response = requests.post(
            url,
            headers=_build_headers("application/vnd.github.v3+json"),
            json={"body": body},
            timeout=GITHUB_HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        _raise_transport_error("post review comment", repo, pr_number, exc)
    except requests.ConnectionError as exc:
        _raise_transport_error("post review comment", repo, pr_number, exc)
    except requests.HTTPError as exc:
        _raise_transport_error("post review comment", repo, pr_number, exc)
    except requests.RequestException as exc:
        _raise_transport_error("post review comment", repo, pr_number, exc)
    return response.json()
