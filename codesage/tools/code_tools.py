import re
import subprocess
import tempfile
import os


def parse_diff(diff_text: str) -> list[dict]:
    chunks = []
    current_file = None
    current_lines = []
    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            if current_file and current_lines:
                chunks.append({"file": current_file, "lines": "\n".join(current_lines)})
            current_file = line.split(" b/")[-1]
            current_lines = []
        elif line.startswith(("+", "-", " ")):
            current_lines.append(line)
    if current_file and current_lines:
        chunks.append({"file": current_file, "lines": "\n".join(current_lines)})
    return chunks


def run_linter(code: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        tmp = f.name
    try:
        result = subprocess.run(
            ["flake8", "--max-line-length=120", tmp],
            capture_output=True, text=True
        )
        output = result.stdout.replace(tmp, "<code>")
        return output.strip() or "No issues found."
    finally:
        os.unlink(tmp)
