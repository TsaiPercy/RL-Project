"""Prompt template 管理 — MiniGrid ASCII grid + JSON 格式規範。

Per SPEC §3 [impl-updated 2026-05-07]: LLM 生成 15×15 ASCII Grid（含外牆）+ JSON
(objects + agent_start)。原本是 13×13 inner-only，現改為完整 15×15 含外牆，
LLM 必須自己生外圍 wall ring，objects/agent 限制在 1-13 inner area。
Per SPEC §11 Module A: prompts.py 管理 prompt template。
"""

from __future__ import annotations

MINIGRID_SYSTEM_PROMPT = """\
You are a game level designer for MiniGrid, a grid-based puzzle game. \
Your task is to design creative and challenging levels."""

MINIGRID_LEVEL_PROMPT = """\
Design a MiniGrid level on a 15x15 grid. The level should be solvable and interesting.

## Rules
- The grid is 15 rows by 15 columns (coordinates 0-14 for both x and y).
- The grid uses two characters: 'W' for wall, '.' for floor.
- The OUTER ring (row 0, row 14, column 0, column 14) MUST all be 'W' (the boundary wall).
- All other tiles (rows 1-13, columns 1-13) are the playable area where you place walls/floors.
- Objects are placed via JSON. Colors available: red, green, blue, purple, yellow, grey.
- The agent's mission is to navigate to a designated object (GoTo task).

## Available Objects
- wall ('W' in grid): Impassable terrain, cannot be walked through.
- floor ('.' in grid): Walkable tile. Can optionally have a color.
- key: A key that can open a door of the same color.
- door: A door tile.
- ball: A movable sphere.

## Output Format

You MUST output exactly this format (no extra text before or after):

Grid:
<15 lines, each exactly 15 characters of 'W' or '.'; outer ring must be all 'W'>

{
  "objects": [
    {"type": "<object_type>", "x": <int>, "y": <int>, "color": "<color>"},
    ...
  ],
  "agent_start": {"x": <int>, "y": <int>, "dir": <0-3>},
  "goal": <int>
}

## Constraints
- "goal" is the 0-based index into the "objects" array, indicating which object the agent must navigate to. It must refer to a non-wall, non-door object (key, ball, or box).
- agent_start direction: 0=right, 1=down, 2=left, 3=up.
- All object and agent_start coordinates must be in range [1, 13] (cannot occupy the outer wall ring).
- Agent and objects cannot overlap with walls or each other.
- The level must be solvable (agent can reach the target object, keys accessible before locked doors).

## Example

Grid:
WWWWWWWWWWWWWWW
W.............W
W..WWW........W
W....W........W
W.............W
W.............W
W.............W
W.............W
W.............W
W.............W
W.............W
W.............W
W.............W
W.............W
WWWWWWWWWWWWWWW

{
  "objects": [
    {"type": "key", "x": 1, "y": 5, "color": "yellow"},
    {"type": "door", "x": 4, "y": 4, "color": "yellow"},
    {"type": "ball", "x": 13, "y": 13, "color": "blue"}
  ],
  "agent_start": {"x": 1, "y": 1, "dir": 0},
  "goal": 2
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
