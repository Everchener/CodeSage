"""技能模块总览。

推荐阅读顺序：
1. 先从 `codesage.api.main.chat` 看起，理解技能命令会在请求刚进入时被解析，
   并在 route 决策后参与自动选择与加载。
2. 回到本文件先看 `parse_skill_command()`，它决定用户是不是显式用了
   `/skill:<name> <task>` 这种命令。
3. 再看 `discover_skills()` 和 `_discover_source_skills()`，理解系统如何从用户目录
   与项目目录发现技能，并且为什么项目技能会覆盖同名用户技能。
4. 然后看 `select_skill()`，它负责在支持的路由里让 LLM 从候选技能中最多选一个。
5. 接着看 `load_skill()`，理解发现到的技能为什么还要经过路径越界、编码和空文件校验。
6. 最后看 `render_skill_prompt_section()`，它决定技能是如何被包装成 prompt 片段，
   再传给 RAG/修改类 agent 的。

执行时序：
- 显式技能链路：用户消息 -> `parse_skill_command` -> `discover_skills`
  -> 按名字定位技能 -> `load_skill` -> `render_skill_prompt_section`
  -> agent 消费技能上下文。
- 自动技能链路：普通消息 -> `discover_skills` -> `select_skill`
  -> `load_skill` -> `render_skill_prompt_section` -> agent 消费技能上下文。

阅读这个文件时，最重要的心智模型是：
- `SkillMetadata` 表示“发现到了一个技能目录，但还没真正读出正文”；
- `SkillSelection` 表示“这个技能被显式指定或自动选中了”；
- `ResolvedSkill` 表示“技能正文已经安全加载完，可以真正喂给模型了”。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from codesage.tools.prompt_tools import build_json_output, build_prompt, build_system_prompt

logger = logging.getLogger(__name__)

# 这些常量定义了技能系统的外部约束：
# - 默认项目根目录；
# - 哪些 route 允许技能参与；
# - `/skill:` 命令与技能名的合法格式；
# - `SKILL.md` frontmatter 的提取方式。
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_SKILL_ROUTES = {"rag", "modify"}
SKILL_COMMAND_PATTERN = re.compile(
    r"^\s*/skill:(?P<name>[a-z0-9]+(?:-[a-z0-9]+)*)"
    r"(?:\s+(?P<request>.*\S))?\s*$",
    flags=re.DOTALL,
)
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", flags=re.DOTALL)

SKILL_SELECTOR_SYSTEM = build_system_prompt(
    role="你是 CodeSage 的技能选择助手，负责在当前请求中判断是否需要启用某个技能。",
    responsibilities=[
        "根据用户请求、当前路由和技能描述选择 0 个或 1 个最合适的技能。",
        "只有当某个技能能明显提升当前任务质量或一致性时才启用它。",
        "如果没有明确匹配，就返回不启用技能。",
    ],
    rules=[
        "最多只能选择一个 skill。",
        "只能从给定的 skill 名单中选择 skill_name。",
        "review、index、none 等不支持的场景必须返回不启用。",
        "reason 使用简洁中文，直接说明是否命中及原因。",
    ],
    output_instruction="只返回 JSON 对象，不要输出 Markdown、解释或思考过程。",
)
SKILL_SOURCE_LABELS = {
    "user": "用户目录",
    "project": "项目目录",
}
SKILL_SELECTION_MODE_LABELS = {
    "explicit": "显式指定",
    "auto": "自动选择",
}


class SkillError(ValueError):
    """技能发现与加载相关错误的基类。"""


class SkillCommandError(SkillError):
    """`/skill:<name>` 命令格式无效时抛出。"""


class SkillNotFoundError(SkillError):
    """请求的技能不存在时抛出。"""


class SkillLoadError(SkillError):
    """请求的技能无法被安全加载时抛出。"""


@dataclass(frozen=True)
class SkillMetadata:
    # 技能名既是用户显式调用时的标识，也是发现阶段去重覆盖时的主键。
    name: str
    # 简要描述会直接暴露给技能选择器，让模型基于描述做自动匹配。
    description: str
    # `path` 指向实际的 `SKILL.md` 文件。
    path: str
    # `source` 标记它来自用户目录还是项目目录，便于调试和覆盖策略解释。
    source: Literal["user", "project"]
    # `root_path` 是允许读取的根目录；后面 `load_skill()` 会用它做路径越界校验。
    root_path: str
    # 允许工具是元数据，不做强校验，但会作为提示词的一部分给到下游 agent。
    allowed_tools: tuple[str, ...] = ()
    compatibility: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        # 统一的序列化出口，方便跨模块传递到 API 层或 agent 层。
        return {
            "name": self.name,
            "description": self.description,
            "path": self.path,
            "source": self.source,
            "allowed_tools": list(self.allowed_tools),
            "compatibility": self.compatibility,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ParsedSkillCommand:
    # 原始消息保留下来，便于错误提示、调试和上层回显。
    original_message: str
    # user_request 是剥掉 `/skill:<name>` 之后真正要交给 agent 的任务文本。
    user_request: str
    # 普通消息时这里为 None；显式命令时这里保存用户点名的技能名。
    skill_name: str | None = None
    # `is_explicit` 是后续分支判断的关键：决定是否必须按指定名字加载技能。
    is_explicit: bool = False


@dataclass(frozen=True)
class SkillSelection:
    # 这里绑定的是“被选中的技能元数据”，还没有读取正文。
    metadata: SkillMetadata
    # mode 区分是用户显式指定，还是系统自动挑出来的。
    mode: Literal["explicit", "auto"]
    # 自动选择时会携带模型给出的简短原因。
    reason: str = ""
    # 保存当前轮用户任务，后面渲染给 agent 时可以一起展示。
    user_request: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "mode": self.mode,
            "reason": self.reason,
            "user_request": self.user_request,
        }


@dataclass(frozen=True)
class ResolvedSkill:
    # metadata 仍然保留技能的来源、路径和描述等发现期信息。
    metadata: SkillMetadata
    # content 是完整的 `SKILL.md` 正文；运行到真正用 skill 的 agent 时最关键的就是它。
    content: str
    user_request: str
    selection_mode: Literal["explicit", "auto"]
    selection_reason: str = ""

    def to_context_dict(self) -> dict[str, Any]:
        # 将“发现期元数据 + 选择结果 + 正文内容”合并成统一上下文字典，
        # 方便不同 agent 不关心具体 dataclass 类型。
        return {
            **self.metadata.to_dict(),
            "content": self.content,
            "user_request": self.user_request,
            "selection_mode": self.selection_mode,
            "selection_reason": self.selection_reason,
        }


def _load_yaml_module():
    # 优先尝试使用 PyYAML；如果运行环境没装，则回退到本文件内的简化解析器。
    try:
        import yaml
    except ImportError:  # pragma: no cover - environment dependent
        return None
    return yaml


def _strip_yaml_scalar(value: str) -> Any:
    # 这个函数只处理最简单的标量归一化，目标不是完整 YAML 语义，
    # 而是给 fallback 解析器提供“够用且可预期”的行为。
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    if text.lower() in {"null", "none"}:
        return None
    return text


def _fallback_yaml_load(frontmatter: str) -> dict[str, Any] | None:
    # 运行到这里说明 PyYAML 不可用，但系统仍希望尽量读取简单 frontmatter。
    # 这个解析器只支持当前技能文件实际会用到的子集：标量、列表和 metadata 映射。
    data: dict[str, Any] = {}
    current_key: str | None = None

    for raw_line in frontmatter.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()

        if indent == 0:
            current_key = None
            if ":" not in stripped:
                # 顶层行连 key:value 结构都不满足，直接判定整个 frontmatter 无效。
                return None
            key, raw_value = stripped.split(":", 1)
            key = key.strip()
            value = raw_value.strip()
            if not key:
                return None
            if value:
                data[key] = _strip_yaml_scalar(value)
            else:
                if key == "metadata":
                    data[key] = {}
                else:
                    data[key] = []
                current_key = key
            continue

        if current_key is None:
            # 有缩进行但没有可挂载的父 key，说明结构不合法。
            return None

        target = data.get(current_key)
        if isinstance(target, list):
            item = stripped[2:].strip() if stripped.startswith("- ") else stripped
            if item:
                target.append(_strip_yaml_scalar(item))
            continue

        if isinstance(target, dict):
            if ":" not in stripped:
                return None
            child_key, child_value = stripped.split(":", 1)
            child_key = child_key.strip()
            if not child_key:
                return None
            target[child_key] = _strip_yaml_scalar(child_value.strip())
            continue

        return None

    return data


def _parse_frontmatter(content: str, skill_path: Path) -> dict[str, Any] | None:
    # 运行到这里说明发现流程已经读取到了某个 `SKILL.md` 文件，
    # 现在要判断它是不是一个结构合法的技能文件。
    match = FRONTMATTER_PATTERN.match(content)
    if not match:
        logger.warning("跳过技能 %s：缺少 YAML frontmatter", skill_path)
        return None

    yaml = _load_yaml_module()
    if yaml is not None:
        try:
            data = yaml.safe_load(match.group(1))
        except Exception:
            logger.warning("跳过技能 %s：YAML frontmatter 无效", skill_path, exc_info=True)
            return None
    else:
        data = _fallback_yaml_load(match.group(1))

    if not isinstance(data, dict):
        logger.warning("跳过技能 %s：frontmatter 不是映射结构", skill_path)
        return None
    return data


def _parse_allowed_tools(raw: object) -> tuple[str, ...]:
    # `allowed-tools` 允许写成空值、字符串或列表，这里统一收敛成 tuple。
    if raw is None:
        return ()
    if isinstance(raw, str):
        items = [item.strip(", ") for item in raw.split() if item.strip(", ")]
        return tuple(items)
    if isinstance(raw, list):
        items = [str(item).strip() for item in raw if str(item).strip()]
        return tuple(items)
    return ()


def _normalize_metadata(raw: object) -> dict[str, str]:
    # metadata 是开放字段，但最终还是要裁成干净的字符串映射，避免脏数据穿透到 prompt。
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, str] = {}
    for key, value in raw.items():
        key_text = str(key).strip()
        value_text = str(value).strip()
        if key_text and value_text:
            normalized[key_text] = value_text
    return normalized


def _resolve_skill_file(skill_file: Path, root_dir: Path) -> Path | None:
    # 这是发现阶段的第一层安全闸门：把路径 resolve 后，确认它仍然待在允许根目录下面。
    try:
        resolved_root = root_dir.resolve()
        resolved_skill = skill_file.resolve()
    except OSError:
        logger.warning("跳过技能 %s：路径解析失败", skill_file, exc_info=True)
        return None

    if not resolved_skill.is_relative_to(resolved_root):
        logger.warning(
            "跳过技能 %s：解析后的路径越过了根目录 %s",
            resolved_skill,
            resolved_root,
        )
        return None
    return resolved_skill


def _discover_source_skills(
    source_dir: Path,
    source: Literal["user", "project"],
) -> list[SkillMetadata]:
    # 这个函数只负责“扫描单个来源目录”，不处理跨来源覆盖。
    if not source_dir.exists() or not source_dir.is_dir():
        return []

    resolved_root = source_dir.resolve()
    discovered: list[SkillMetadata] = []

    for child in sorted(source_dir.iterdir(), key=lambda item: item.name.lower()):
        # 运行到这里说明发现流程正在逐个遍历 `.agents/skills` 下的候选目录。
        if not child.is_dir():
            continue

        skill_file = child / "SKILL.md"
        if not skill_file.exists() or not skill_file.is_file():
            continue

        resolved_skill = _resolve_skill_file(skill_file, resolved_root)
        if resolved_skill is None:
            continue

        try:
            # 这里强制 UTF-8，和项目整体编码规范保持一致。
            content = resolved_skill.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("跳过技能 %s：文件不是 UTF-8 编码", resolved_skill)
            continue
        except OSError:
            logger.warning("跳过技能 %s：读取失败", resolved_skill, exc_info=True)
            continue

        frontmatter = _parse_frontmatter(content, resolved_skill)
        if frontmatter is None:
            continue

        name = str(frontmatter.get("name", "")).strip()
        description = str(frontmatter.get("description", "")).strip()

        if not name or not description:
            logger.warning("跳过技能 %s：缺少 name 或 description", resolved_skill)
            continue
        if not SKILL_NAME_PATTERN.fullmatch(name):
            # 技能名限制成小写/kebab-case，是为了让显式命令和目录结构都保持稳定可预测。
            logger.warning("跳过技能 %s：技能名称 %r 不合法", resolved_skill, name)
            continue
        if child.name != name:
            # 目录名和 frontmatter 名称必须一致，避免“目录 A 里伪装成技能 B”的混淆。
            logger.warning(
                "跳过技能 %s：目录名 %r 与技能名称 %r 不一致",
                resolved_skill,
                child.name,
                name,
            )
            continue

        discovered.append(
            SkillMetadata(
                name=name,
                description=description,
                path=str(resolved_skill),
                source=source,
                root_path=str(resolved_root),
                allowed_tools=_parse_allowed_tools(frontmatter.get("allowed-tools")),
                compatibility=str(frontmatter.get("compatibility", "")).strip() or None,
                metadata=_normalize_metadata(frontmatter.get("metadata")),
            )
        )

    return discovered


def discover_skills(project_root: str | Path | None = None) -> list[SkillMetadata]:
    """从 deepagents 风格的 `.agents/skills` 目录发现本地技能。"""
    # 运行到这里通常发生在 `/chat` 请求准备启用显式/自动技能之前。
    # 系统会同时扫描：
    # - 用户目录：`~/.agents/skills`
    # - 项目目录：`<project>/.agents/skills`
    resolved_project_root = Path(project_root or DEFAULT_PROJECT_ROOT).resolve()
    user_skills_dir = Path.home() / ".agents" / "skills"
    project_skills_dir = resolved_project_root / ".agents" / "skills"

    merged: dict[str, SkillMetadata] = {}
    for source_dir, source in (
        (user_skills_dir, "user"),
        (project_skills_dir, "project"),
    ):
        for metadata in _discover_source_skills(source_dir, source):
            # 后写入的项目技能会覆盖先写入的同名用户技能；
            # 这是一个有意的优先级设计，让仓库可以对本项目的技能版本做定制。
            merged[metadata.name] = metadata
    return sorted(merged.values(), key=lambda item: item.name)


def parse_skill_command(message: str) -> ParsedSkillCommand:
    """解析 `/skill:<name> <request>`，普通消息则保持原样。"""
    # 这是技能链路的第一步。运行到这里时，请求还没进入 route 判定，
    # 系统只是先判断用户有没有显式点名某个技能。
    stripped = (message or "").strip()
    if not stripped.lower().startswith("/skill:"):
        # 普通消息直接原样透传，让后续流程决定是否自动选 skill。
        return ParsedSkillCommand(
            original_message=message,
            user_request=message,
            skill_name=None,
            is_explicit=False,
        )

    match = SKILL_COMMAND_PATTERN.match(message or "")
    if not match:
        raise SkillCommandError("技能命令格式无效，请使用 `/skill:<name> <用户任务>`。")

    skill_name = match.group("name")
    user_request = (match.group("request") or "").strip()
    if not user_request:
        raise SkillCommandError("`/skill:<name>` 后还需要提供用户任务。")

    return ParsedSkillCommand(
        original_message=message,
        user_request=user_request,
        skill_name=skill_name,
        is_explicit=True,
    )


def _skill_list_text(skills: Sequence[SkillMetadata]) -> str:
    # 把结构化技能列表压成给 LLM 看的候选清单文本。
    lines: list[str] = []
    for skill in skills:
        description = f"- `{skill.name}`: {skill.description}"
        annotations: list[str] = []
        if skill.compatibility:
            annotations.append(f"兼容性: {skill.compatibility}")
        if skill.allowed_tools:
            annotations.append(f"建议工具: {', '.join(skill.allowed_tools)}")
        if annotations:
            description += f" ({'; '.join(annotations)})"
        source_label = SKILL_SOURCE_LABELS.get(skill.source, skill.source)
        description += f" [来源={source_label}]"
        lines.append(description)
    return "\n".join(lines)


def _selector_call(
    selector: Callable[..., dict[str, Any] | list[Any] | None],
    prompt: str,
    system: str,
) -> dict[str, Any] | list[Any] | None:
    # 兼容两类 selector：
    # - 标准 `call_llm_json(prompt, system=..., max_tokens=...)`
    # - 测试里只接受一个 prompt 参数的假函数。
    try:
        return selector(prompt, system=system, max_tokens=250)
    except TypeError:
        return selector(prompt)


def select_skill(
    *,
    user_input: str,
    route: str,
    skills: Sequence[SkillMetadata],
    selector: Callable[..., dict[str, Any] | list[Any] | None] | None = None,
) -> SkillSelection | None:
    """为当前路由和用户请求最多选择一个技能。"""
    # 运行到这里说明：
    # - 用户没有显式指定技能，或者系统仍想在支持的 route 里做自动补强；
    # - 发现流程已经拿到可用技能清单。
    if route not in SUPPORTED_SKILL_ROUTES or not skills:
        # 明确禁止在不支持的 route 上启用技能，避免把 review/index 等路径复杂化。
        return None
    if selector is None:
        from codesage.tools.llm_tools import call_llm_json

        selector = call_llm_json

    # 这里把技能选择问题显式收束成一个 JSON 决策任务：
    # 要么选一个已知技能，要么明确返回“不启用”。
    prompt = build_prompt(
        task="判断当前请求是否需要启用某个技能，如果需要则只返回一个最合适的技能。",
        context_sections=[
            ("当前路由", route),
            ("用户请求", user_input),
            ("可用技能", _skill_list_text(skills)),
        ],
        rules=[
            "只有在某个技能的描述与当前任务高度匹配时才启用。",
            "如果没有明确匹配，返回 use_skill=false。",
            "不能臆造技能名称，只能从给定名单中选择。",
        ],
        output_format=build_json_output(
            """{
  "use_skill": true,
  "skill_name": "example-skill",
  "reason": "简短中文原因"
}""",
            extra_rules=[
                "如果不启用 skill，返回 `{\"use_skill\": false, \"skill_name\": \"\", \"reason\": \"...\"}`。",
            ],
        ),
    )

    payload = _selector_call(selector, prompt, SKILL_SELECTOR_SYSTEM)
    if not isinstance(payload, dict):
        return None

    if not payload.get("use_skill"):
        # 运行到这里说明选择器明确认为“这次请求不需要 skill”。
        return None

    skill_name = str(payload.get("skill_name", "")).strip()
    if not skill_name:
        return None

    selected = next((item for item in skills if item.name == skill_name), None)
    if selected is None:
        # 即使模型说要启用，也必须落在候选清单里，否则判为无效结果。
        logger.warning("技能选择器返回了未知的 skill %r", skill_name)
        return None

    return SkillSelection(
        metadata=selected,
        mode="auto",
        reason=str(payload.get("reason", "")).strip(),
        user_request=user_input,
    )


def load_skill(selection: SkillSelection | SkillMetadata) -> ResolvedSkill:
    """加载并校验已发现技能的完整 `SKILL.md` 内容。"""
    # 发现到技能元数据之后，仍然不能直接把路径交给 agent。
    # 这里是真正的“安全加载”步骤。
    if isinstance(selection, SkillSelection):
        metadata = selection.metadata
        selection_mode = selection.mode
        selection_reason = selection.reason
        user_request = selection.user_request
    else:
        metadata = selection
        selection_mode = "explicit"
        selection_reason = ""
        user_request = ""

    skill_path = Path(metadata.path)
    root_path = Path(metadata.root_path)
    try:
        # 再做一次 resolve，是为了防止发现后到加载前路径发生变化或符号链接绕过。
        resolved_skill = skill_path.resolve()
        resolved_root = root_path.resolve()
    except OSError as exc:
        raise SkillLoadError(f"技能 `{metadata.name}` 的路径解析失败：{exc}") from exc

    if not resolved_skill.is_relative_to(resolved_root):
        # 运行到这里说明技能路径已经越出允许根目录，必须拒绝读取。
        raise SkillLoadError(
            f"技能 `{metadata.name}` 的路径越界，拒绝读取：{resolved_skill}"
        )

    try:
        # 统一要求 UTF-8，既符合仓库规范，也避免 prompt 里出现乱码。
        content = resolved_skill.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SkillLoadError(
            f"技能 `{metadata.name}` 的 SKILL.md 不是 UTF-8 编码。"
        ) from exc
    except OSError as exc:
        raise SkillLoadError(f"技能 `{metadata.name}` 读取失败：{exc}") from exc

    if not content.strip():
        raise SkillLoadError(f"技能 `{metadata.name}` 的 SKILL.md 为空。")

    # 运行到这里说明这个技能已经从“发现到的候选项”升级成“可安全使用的技能上下文”。
    return ResolvedSkill(
        metadata=metadata,
        content=content,
        user_request=user_request,
        selection_mode=selection_mode,
        selection_reason=selection_reason,
    )


def render_skill_prompt_section(
    skill_context: ResolvedSkill | dict[str, Any] | None,
    *,
    include_full_content: bool,
) -> str:
    """为下游提示词渲染结构化的技能上下文区块。"""
    # 这是技能模块对 agent 侧的最后一步输出。
    # 运行到这里时，RAG agent 或代码修改 agent 已经准备构造 prompt，
    # 它只需要一段规范的文本区块，而不想关心技能对象来自哪种 dataclass。
    if skill_context is None:
        return ""

    if isinstance(skill_context, ResolvedSkill):
        normalized = skill_context.to_context_dict()
    else:
        normalized = dict(skill_context)

    name = str(normalized.get("name", "")).strip()
    description = str(normalized.get("description", "")).strip()
    source = str(normalized.get("source", "")).strip()
    compatibility = str(normalized.get("compatibility", "")).strip()
    content = str(normalized.get("content", "")).strip()
    user_request = str(normalized.get("user_request", "")).strip()
    selection_mode = str(normalized.get("selection_mode", "")).strip()
    allowed_tools = normalized.get("allowed_tools", [])
    tool_list = ", ".join(str(item).strip() for item in allowed_tools if str(item).strip())
    source_label = SKILL_SOURCE_LABELS.get(source, source)
    selection_mode_label = SKILL_SELECTION_MODE_LABELS.get(selection_mode, selection_mode)

    lines = ["## 已启用技能指南"]
    # 这里先渲染“摘要信息”，让模型先知道技能是什么、为何被启用、建议使用哪些工具。
    if name:
        lines.append(f"- 名称: `{name}`")
    if description:
        lines.append(f"- 作用: {description}")
    if source_label:
        lines.append(f"- 来源: {source_label}")
    if selection_mode_label:
        lines.append(f"- 启用方式: {selection_mode_label}")
    if compatibility:
        lines.append(f"- 兼容性: {compatibility}")
    if tool_list:
        lines.append(f"- 建议工具: {tool_list}")
    if user_request:
        lines.append(f"- 本轮用户任务: {user_request}")

    if include_full_content and content:
        # rewrite 阶段通常不需要全量正文，只带摘要；
        # 生成或改码阶段才会把完整 `SKILL.md` 指令拼进去。
        lines.extend(["", "### SKILL.md 全量指令", content])

    return "\n".join(lines).strip()


__all__ = [
    "ParsedSkillCommand",
    "ResolvedSkill",
    "SUPPORTED_SKILL_ROUTES",
    "SkillCommandError",
    "SkillLoadError",
    "SkillMetadata",
    "SkillNotFoundError",
    "SkillSelection",
    "discover_skills",
    "load_skill",
    "parse_skill_command",
    "render_skill_prompt_section",
    "select_skill",
]
