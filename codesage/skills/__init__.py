"""技能发现与执行辅助工具。"""

from codesage.skills.discovery import (
    ParsedSkillCommand,
    ResolvedSkill,
    SUPPORTED_SKILL_ROUTES,
    SkillCommandError,
    SkillLoadError,
    SkillMetadata,
    SkillNotFoundError,
    SkillSelection,
    discover_skills,
    load_skill,
    parse_skill_command,
    render_skill_prompt_section,
    select_skill,
)
from codesage.skills.registry import (
    SkillExecutionSpec,
    build_skill_execution_task,
    collect_skill_cache_metadata,
    normalize_skill_args,
    resolve_skill_execution,
)

__all__ = [
    "ParsedSkillCommand",
    "ResolvedSkill",
    "SUPPORTED_SKILL_ROUTES",
    "SkillCommandError",
    "SkillExecutionSpec",
    "SkillLoadError",
    "SkillMetadata",
    "SkillNotFoundError",
    "SkillSelection",
    "build_skill_execution_task",
    "collect_skill_cache_metadata",
    "discover_skills",
    "load_skill",
    "normalize_skill_args",
    "parse_skill_command",
    "render_skill_prompt_section",
    "resolve_skill_execution",
    "select_skill",
]
