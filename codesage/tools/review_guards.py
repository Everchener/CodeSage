from __future__ import annotations

import hashlib
import hmac
import re

from codesage.tools.code_tools import parse_diff


class WebhookAuthError(ValueError):
    """Raised when a webhook request fails signature validation."""


class ReviewInputError(ValueError):
    """Raised when review input does not satisfy the required contract."""


_REPO_PATTERN = re.compile(r"^[^/\s]+/[^/\s]+$")


def verify_github_webhook_signature(
    secret: str,
    raw_body: bytes,
    signature_header: str | None,
) -> None:
    if not signature_header or not signature_header.startswith("sha256="):
        raise WebhookAuthError("Invalid webhook signature.")

    expected_signature = "sha256=" + hmac.new(
        secret.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected_signature, signature_header.strip()):
        raise WebhookAuthError("Invalid webhook signature.")


def validate_repo_and_pr(repo: str, pr_number: int) -> None:
    if not _REPO_PATTERN.fullmatch(repo or ""):
        raise ReviewInputError("repo must be in owner/name format.")
    if pr_number <= 0:
        raise ReviewInputError("pr_number must be greater than 0.")


def validate_review_diff(diff_text: str, max_bytes: int) -> str:
    normalized_diff = (diff_text or "").strip()
    if not normalized_diff:
        raise ReviewInputError("diff_text cannot be empty.")

    diff_bytes = normalized_diff.encode("utf-8")
    if len(diff_bytes) > max_bytes:
        raise ReviewInputError(f"diff_text exceeds max size of {max_bytes} bytes.")

    if "diff --git " not in normalized_diff:
        raise ReviewInputError("diff_text must contain at least one diff --git header.")

    if not parse_diff(normalized_diff):
        raise ReviewInputError("diff_text could not be parsed as a Git diff.")

    return normalized_diff
