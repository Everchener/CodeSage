from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

MAX_DISCOVERED_TEST_TARGETS = 8
MAX_REPORT_OUTPUT_CHARS = 4000
SANDBOX_RUNS_DIRNAME = "sandbox_test_runs"
SANDBOX_SKIP_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    ".venv",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".codesage",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    "coverage",
    "htmlcov",
    ".chrome-headless",
    "deepagents-main",
}
SANDBOX_SKIP_PREFIXES = (
    ".tmp",
    "pytest-cache-files-",
    "skill-test-",
    "pytest_run_",
)
TOKEN_STOPWORDS = {
    "agent",
    "agents",
    "api",
    "app",
    "code",
    "codesage",
    "file",
    "files",
    "main",
    "module",
    "modify",
    "py",
    "python",
    "service",
    "test",
    "tests",
    "tool",
    "tools",
    "unit",
    "utils",
}


@dataclass(frozen=True)
class SandboxTestResult:
    status: str
    sandbox_dir: str
    command: list[str]
    test_targets: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    reason: str = ""


def _should_skip_name(name: str) -> bool:
    return name in SANDBOX_SKIP_NAMES or any(name.startswith(prefix) for prefix in SANDBOX_SKIP_PREFIXES)


def _ignore_copy(_src: str, names: list[str]) -> list[str]:
    return [name for name in names if _should_skip_name(name)]


def _normalize_rel_path(raw_path: str, working_dir: str) -> str:
    root = Path(working_dir).resolve()
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = (root / candidate).resolve()
    try:
        relative = candidate.relative_to(root)
        return relative.as_posix()
    except ValueError:
        return raw_path.replace("\\", "/").lstrip("./")


def _module_hint(path_text: str) -> str:
    path = Path(path_text)
    parts = list(path.parts)
    if path.suffix == ".py":
        parts[-1] = path.stem
    return ".".join(part for part in parts if part and part not in {".", ".."})


def _collect_tokens(paths: list[str]) -> set[str]:
    tokens: set[str] = set()
    for path_text in paths:
        for token in re.split(r"[^a-z0-9_]+", path_text.lower()):
            if len(token) < 3 or token in TOKEN_STOPWORDS:
                continue
            tokens.add(token)
    return tokens


def discover_related_test_targets(working_dir: str, changed_paths: list[str]) -> list[str]:
    root = Path(working_dir).resolve()
    tests_dir = root / "tests"
    if not tests_dir.is_dir():
        return []

    normalized_paths = [_normalize_rel_path(path, working_dir) for path in changed_paths if str(path).strip()]
    explicit_tests = sorted(
        {
            path
            for path in normalized_paths
            if path.startswith("tests/") and Path(root / path).is_file()
        }
    )
    if explicit_tests:
        return explicit_tests

    content_hints = {
        hint.lower()
        for path in normalized_paths
        for hint in (path, _module_hint(path), Path(path).stem)
        if hint
    }
    keyword_tokens = _collect_tokens(normalized_paths)
    ranked: list[tuple[int, str]] = []

    for test_file in sorted(tests_dir.rglob("test_*.py")):
        if not test_file.is_file():
            continue
        rel_path = test_file.relative_to(root).as_posix()
        stem = test_file.stem.lower()
        text = test_file.read_text(encoding="utf-8", errors="ignore").lower()

        score = 0
        if any(hint and hint in rel_path.lower() for hint in content_hints):
            score += 3
        if any(hint and hint in stem for hint in content_hints):
            score += 2
        if any(hint and hint in text for hint in content_hints):
            score += 6
        if keyword_tokens:
            score += sum(1 for token in keyword_tokens if token in rel_path.lower() or token in stem)
            if any(token in text for token in keyword_tokens):
                score += 2

        if score > 0:
            ranked.append((score, rel_path))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [path for _, path in ranked[:MAX_DISCOVERED_TEST_TARGETS]]


def _create_sandbox_workspace(working_dir: str, run_label: str) -> Path:
    root = Path(working_dir).resolve()
    sandbox_base = root / SANDBOX_RUNS_DIRNAME
    sandbox_base.mkdir(parents=True, exist_ok=True)
    sandbox_dir = Path(tempfile.mkdtemp(prefix=f"{run_label}_", dir=str(sandbox_base)))

    for child in root.iterdir():
        if _should_skip_name(child.name):
            continue
        destination = sandbox_dir / child.name
        if child.is_dir():
            shutil.copytree(child, destination, ignore=_ignore_copy)
        else:
            shutil.copy2(child, destination)

    return sandbox_dir


def run_pytest_in_sandbox(
    working_dir: str,
    test_targets: list[str],
    *,
    timeout_seconds: int = 120,
    run_label: str = "pytest_run",
) -> SandboxTestResult:
    unique_targets = []
    seen: set[str] = set()
    for target in test_targets:
        normalized = target.replace("\\", "/").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_targets.append(normalized)

    if not unique_targets:
        return SandboxTestResult(
            status="skipped",
            sandbox_dir="",
            command=[],
            test_targets=[],
            returncode=None,
            stdout="",
            stderr="",
            reason="未发现可执行的关联测试。",
        )

    sandbox_dir = _create_sandbox_workspace(working_dir, run_label)
    command = [sys.executable, "-m", "pytest", *unique_targets, "-q"]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    try:
        completed = subprocess.run(
            command,
            cwd=str(sandbox_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            env=env,
        )
        return SandboxTestResult(
            status="passed" if completed.returncode == 0 else "failed",
            sandbox_dir=str(sandbox_dir),
            command=command,
            test_targets=unique_targets,
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            reason="沙箱 pytest 已执行完成。",
        )
    except subprocess.TimeoutExpired as exc:
        return SandboxTestResult(
            status="timed_out",
            sandbox_dir=str(sandbox_dir),
            command=command,
            test_targets=unique_targets,
            returncode=None,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            reason=f"沙箱 pytest 超时（>{timeout_seconds}s）。",
        )
    except Exception as exc:  # pragma: no cover - defensive
        return SandboxTestResult(
            status="error",
            sandbox_dir=str(sandbox_dir),
            command=command,
            test_targets=unique_targets,
            returncode=None,
            stdout="",
            stderr=str(exc),
            reason="沙箱 pytest 执行失败。",
        )


def _trim_output(text: str) -> str:
    stripped = text.strip()
    if len(stripped) <= MAX_REPORT_OUTPUT_CHARS:
        return stripped
    return f"{stripped[:MAX_REPORT_OUTPUT_CHARS].rstrip()}\n...<truncated>"


def format_sandbox_test_report(result: SandboxTestResult) -> str:
    if result.status == "skipped":
        return f"沙箱测试已跳过：{result.reason}"

    title_map = {
        "passed": "沙箱测试通过",
        "failed": "沙箱测试未通过",
        "timed_out": "沙箱测试超时",
        "error": "沙箱测试执行失败",
    }
    title = title_map.get(result.status, "沙箱测试状态未知")
    targets = ", ".join(result.test_targets) if result.test_targets else "无"
    parts = [
        f"{title}：{targets}",
        f"沙箱目录：{result.sandbox_dir or '无'}",
    ]
    if result.returncode is not None:
        parts.append(f"返回码：{result.returncode}")
    if result.reason:
        parts.append(f"说明：{result.reason}")

    output = "\n".join(item for item in (_trim_output(result.stdout), _trim_output(result.stderr)) if item)
    if output:
        parts.append("输出：")
        parts.append(output)
    return "\n".join(parts)
