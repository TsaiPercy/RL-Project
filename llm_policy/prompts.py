"""Prompt template 管理 — MiniGrid ASCII grid + JSON 格式規範。

Per SPEC §3: LLM 生成的關卡描述包含 ASCII Grid (13×13) + JSON (objects + agent_start)。
Per SPEC §11 Module A: prompts.py 管理 prompt template。
"""

from __future__ import annotations

MINIGRID_SYSTEM_PROMPT = """\
You are a game level designer for MiniGrid, a grid-based puzzle game. \
Your task is to design creative and challenging levels."""

MINIGRID_LEVEL_PROMPT = """\
Design a MiniGrid level on a 13x13 grid. The level should be solvable and interesting.

## Rules
- The grid is 13 rows by 13 columns (coordinates 0-12 for both x and y).
- The grid uses two characters: 'W' for wall, '.' for floor.
- Objects are placed via JSON. Colors available: red, green, blue, purple, yellow, grey.
- The agent must be able to reach the goal.
- No need to build the outter walls; they are pre-defined, meaning the grid is acutually 15*15 with an outter walls.

## Available Objects
- wall ('W' in grid): Impassable terrain, cannot be walked through.
- floor ('.' in grid): Walkable tile. Can optionally have a color.
- goal: The finish point. The agent wins by reaching it. Exactly one required.
- door: A door tile. States: open / closed / locked.
  - A locked door requires a key of the same color to unlock.
  - A closed (unlocked) door can be opened by the agent directly.
- key: A pickable item. Used to unlock a locked door of the matching color.
- ball: A pickable item. Can be picked up or pushed to an adjacent empty cell.
- box: A container that can be opened. May contain a key inside.

## Output Format

You MUST output exactly this format (no extra text before or after):

Grid:
<13 lines, each exactly 13 characters of 'W' or '.'>

{
  "objects": [
    {"type": "<object_type>", "x": <int>, "y": <int>, "color": "<color>"},
    ...
  ],
  "agent_start": {"x": <int>, "y": <int>, "dir": <0-3>}
}

## Constraints
- Exactly one "goal" object is required.
- agent_start direction: 0=right, 1=down, 2=left, 3=up.
- All coordinates must be in range [0, 12].
- Objects cannot overlap with walls or each other.
- The level must be solvable (agent can reach goal, keys accessible before locked doors).
- If a box contains a key, specify it as: {"type": "box", "x": 3, "y": 5, "color": "red", "contains": {"type": "key", "color": "blue"}}.

## Example

Grid:
.............
.............
..WWW........
....W........
.............
.............
.............
.............
.............
.............
.............
.............
.............

{
  "objects": [
    {"type": "key", "x": 1, "y": 5, "color": "yellow"},
    {"type": "door", "x": 4, "y": 4, "color": "yellow"},
    {"type": "goal", "x": 12, "y": 12}
  ],
  "agent_start": {"x": 0, "y": 0, "dir": 0}
}

Now design a new level:"""


def get_minigrid_prompt() -> str:
    """取得 MiniGrid 關卡生成的 user prompt。

    Returns:
        完整的 prompt 字串（包含格式規範 + 範例）。
    """
    return MINIGRID_LEVEL_PROMPT


def get_system_prompt() -> str:
    """取得 system prompt。

    Returns:
        System role 的 prompt 字串。
    """
    return MINIGRID_SYSTEM_PROMPT


def format_chat_messages(
    system_prompt: str,
    user_prompt: str,
) -> list[dict[str, str]]:
    """將 system + user prompt 組裝為 chat message 格式。

    Args:
        system_prompt: System role 內容。
        user_prompt: User role 內容。

    Returns:
        適合 transformers tokenizer.apply_chat_template() 的 message list。
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
