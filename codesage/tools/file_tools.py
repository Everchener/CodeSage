"""文件操作工具集，供 CodeModifyAgent 使用。"""
import os
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool


class FileOperationError(Exception):
    """文件操作错误"""
    pass


def _validate_path(path: str, working_dir: str = ".") -> Path:
    """验证路径安全，防止路径逃逸。"""
    abs_workdir = Path(working_dir).resolve()
    abs_path = (abs_workdir / path).resolve()

    # 确保路径在工作目录内
    try:
        abs_path.relative_to(abs_workdir)
    except ValueError:
        raise FileOperationError(f"路径不允许超出工作目录: {path}")

    return abs_path


@tool
def read_file(path: str, working_dir: str = ".") -> str:
    """读取指定文件的内容。

    Args:
        path: 文件路径（相对于 working_dir）
        working_dir: 工作目录根路径

    Returns:
        文件内容字符串
    """
    try:
        abs_path = _validate_path(path, working_dir)
        if not abs_path.exists():
            return f"文件不存在: {path}"
        if not abs_path.is_file():
            return f"不是文件: {path}"

        content = abs_path.read_text(encoding="utf-8")
        return f"=== {path} ===\n{content}"
    except FileOperationError as e:
        return str(e)
    except Exception as e:
        return f"读取文件失败: {e}"


@tool
def write_file(path: str, content: str, working_dir: str = ".") -> str:
    """覆盖写入文件（会覆盖原内容）。

    Args:
        path: 文件路径（相对于 working_dir）
        content: 要写入的内容
        working_dir: 工作目录根路径

    Returns:
        操作结果消息
    """
    try:
        abs_path = _validate_path(path, working_dir)

        # 确保父目录存在
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        abs_path.write_text(content, encoding="utf-8")
        return f"✅ 已写入文件: {path}"
    except FileOperationError as e:
        return str(e)
    except Exception as e:
        return f"写入文件失败: {e}"


@tool
def create_file(path: str, content: str, working_dir: str = ".") -> str:
    """创建新文件（如果文件已存在则失败）。

    Args:
        path: 文件路径（相对于 working_dir）
        content: 要写入的内容
        working_dir: 工作目录根路径

    Returns:
        操作结果消息
    """
    try:
        abs_path = _validate_path(path, working_dir)

        if abs_path.exists():
            return f"文件已存在: {path}"

        # 确保父目录存在
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        abs_path.write_text(content, encoding="utf-8")
        return f"✅ 已创建文件: {path}"
    except FileOperationError as e:
        return str(e)
    except Exception as e:
        return f"创建文件失败: {e}"


@tool
def list_files(directory: str = ".", pattern: str = "*", working_dir: str = ".") -> str:
    """列出目录下的文件。

    Args:
        directory: 目录路径（相对于 working_dir）
        pattern: 文件匹配模式（如 "*.py"）
        working_dir: 工作目录根路径

    Returns:
        文件列表字符串
    """
    try:
        abs_dir = _validate_path(directory, working_dir)
        if not abs_dir.exists():
            return f"目录不存在: {directory}"
        if not abs_dir.is_dir():
            return f"不是目录: {directory}"

        files = list(abs_dir.glob(pattern))
        if not files:
            return f"目录 {directory} 中没有匹配 {pattern} 的文件"

        result = [f"📁 {directory}/"]
        for f in sorted(files):
            rel_path = f.relative_to(abs_dir)
            if f.is_dir():
                result.append(f"  📂 {rel_path}/")
            else:
                result.append(f"  📄 {rel_path}")

        return "\n".join(result)
    except FileOperationError as e:
        return str(e)
    except Exception as e:
        return f"列出文件失败: {e}"


@tool
def run_linter(path: str, working_dir: str = ".") -> str:
    """运行 flake8 对文件进行语法检查。

    Args:
        path: 文件路径（相对于 working_dir）
        working_dir: 工作目录根路径

    Returns:
        flake8 检查结果
    """
    try:
        abs_path = _validate_path(path, working_dir)
        if not abs_path.exists():
            return f"文件不存在: {path}"

        # 只检查 .py 文件
        if abs_path.suffix != ".py":
            return f"只支持 .py 文件的语法检查: {path}"

        import subprocess
        result = subprocess.run(
            ["flake8", str(abs_path), "--max-line-length=120", "--ignore=E501,W503"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            return "✅ flake8 检查通过，无语法错误"
        return f"⚠️ flake8 检查结果:\n{result.stdout}{result.stderr}"
    except FileOperationError as e:
        return str(e)
    except subprocess.TimeoutExpired:
        return "flake8 检查超时"
    except Exception as e:
        return f"运行 flake8 失败: {e}"
