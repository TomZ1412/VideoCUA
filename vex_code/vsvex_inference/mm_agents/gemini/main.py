import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .actions import parse_base_response, py_string
from .client import call_gemini
from .history import (
    build_messages,
    dump_debug_messages,
    ensure_empty_think_prefix,
    previous_actions_text,
    update_folding_state,
)
from .images import create_video_clip, encode_video_base64
from .images import image_size_from_base64, image_size_from_bytes, process_image, process_image_resize
from .images import adjust_coordinates
from .parser import extract_action_line, iter_tool_call_params, parse_coordinate, parse_keys, parse_number
from .prompts import build_base_system_prompt, build_base_tools_def, build_instruction_prompt
from .scaffolds import get_gemini_scaffold_profile


logger = None

SEARCH_ACTION_DESCRIPTION_PROMPT = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `key_down`: Press and hold a single key without releasing it.
* `key_up`: Release a previously held single key.
* `left_mouse_down`: Press and hold the left mouse button.
* `left_mouse_up`: Release the left mouse button.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button.
* `middle_click`: Click the middle mouse button.
* `double_click`: Double-click the left mouse button.
* `triple_click`: Triple-click the left mouse button.
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll.
* `screenshot`: Capture a new screenshot of the current screen.
* `wait`: Wait specified seconds for the change to happen.
* `save_knowledge`: Save a task-relevant knowledge item grounded in the currently visible page or screenshot.
* `stop_search`: Stop the search. Use `status=success` only when enough knowledge has already been saved. Use `status=failure` when evidence is insufficient or the search is stuck.
""".strip()

SCROLL_PIXELS_PER_CLICK = 120.0
SCROLL_MAX_CLICKS = 8
SEARCH_ACTION_ENUM = {
    "key",
    "key_down",
    "key_up",
    "left_mouse_down",
    "left_mouse_up",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "triple_click",
    "scroll",
    "hscroll",
    "screenshot",
    "wait",
    "save_knowledge",
    "stop_search",
}
SEARCH_ACTION_ALIASES = {
    "click": "left_click",
    "leftclick": "left_click",
    "doubleclick": "double_click",
    "tripleclick": "triple_click",
    "rightclick": "right_click",
    "middleclick": "middle_click",
    "mousemove": "mouse_move",
    "move_mouse": "mouse_move",
    "drag": "left_click_drag",
    "drag_to": "left_click_drag",
    "leftdrag": "left_click_drag",
    "leftclickdrag": "left_click_drag",
    "keydown": "key_down",
    "keyup": "key_up",
    "mousedown": "left_mouse_down",
    "mouseup": "left_mouse_up",
    "leftmousedown": "left_mouse_down",
    "leftmouseup": "left_mouse_up",
    "saveknowledge": "save_knowledge",
    "save_knowledge_item": "save_knowledge",
    "stopsearch": "stop_search",
    "stop_search_success": "stop_search",
    "stop_search_failure": "stop_search",
}


def _pixels_to_clicks(pixels: int) -> int:
    if pixels == 0:
        return 0
    magnitude = min(
        SCROLL_MAX_CLICKS,
        max(1, int(abs(pixels) / SCROLL_PIXELS_PER_CLICK + 0.5)),
    )
    return magnitude if pixels > 0 else -magnitude


def _normalize_search_action(raw_action: object) -> str:
    action = str(raw_action or "").strip()
    action = action.strip("`'\" \t\r\n")
    action = action.lower().replace("-", "_").replace(" ", "_")
    action = SEARCH_ACTION_ALIASES.get(action, action)
    return action if action in SEARCH_ACTION_ENUM else ""


def build_search_tools_def(processed_width: int, processed_height: int, coordinate_type: str) -> Dict:
    resolution = (
        f"* The screen's resolution is {processed_width}x{processed_height}."
        if coordinate_type == "absolute"
        else "* The screen's resolution is 1000x1000."
    )
    description = "\n".join(
        [
            "Use a mouse and keyboard to interact with a computer, and save structured knowledge items while searching.",
            "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
            "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.",
            resolution,
            "* Save knowledge only when it is grounded in currently visible content from the screenshot or page you have opened.",
            "* If you are not yet ready to stop, keep browsing after saving knowledge.",
        ]
    )
    return {
        "type": "function",
        "function": {
            "name": "computer_use",
            "description": description,
            "parameters": {
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {
                        "type": "string",
                        "description": SEARCH_ACTION_DESCRIPTION_PROMPT,
                        "enum": [
                            "key",
                            "key_down",
                            "key_up",
                            "left_mouse_down",
                            "left_mouse_up",
                            "type",
                            "mouse_move",
                            "left_click",
                            "left_click_drag",
                            "right_click",
                            "middle_click",
                            "double_click",
                            "triple_click",
                            "scroll",
                            "hscroll",
                            "screenshot",
                            "wait",
                            "save_knowledge",
                            "stop_search",
                        ],
                    },
                    "keys": {"type": "array", "description": "Required only by `action=key`, `action=key_down`, or `action=key_up`."},
                    "text": {"type": "string", "description": "Required only by `action=type`."},
                    "coordinate": {
                        "type": "array",
                        "description": "(x, y) coordinates. Required only by `action=mouse_move` and `action=left_click_drag`, optional for `action=left_mouse_down` and `action=left_mouse_up`.",
                    },
                    "pixels": {"type": "number", "description": "Scroll amount. Required only by `action=scroll` or `action=hscroll`."},
                    "time": {"type": "number", "description": "Seconds to wait. Required only by `action=wait`."},
                    "summary": {"type": "string", "description": "Required by `action=save_knowledge`. One concise factual statement."},
                    "steps": {"type": "array", "description": "Optional by `action=save_knowledge`. Concrete actionable steps grounded in what is visible now."},
                    "source": {"type": "string", "description": "Optional by `action=save_knowledge`. The current page URL, site, or title."},
                    "evidence": {"type": "string", "description": "Required by `action=save_knowledge`. A short quote or visible UI labels from the current screenshot/page."},
                    "task_relevance": {"type": "string", "description": "Optional by `action=save_knowledge`. Which part of the original task this helps with."},
                    "uncertainty": {"type": "string", "description": "Optional by `action=save_knowledge`. Specific remaining uncertainty or `none`."},
                    "confidence": {
                        "type": "string",
                        "description": "Optional by `action=save_knowledge`. Confidence in this knowledge item.",
                        "enum": ["high", "medium", "low"],
                    },
                    "reason": {"type": "string", "description": "Optional by `action=stop_search`. Short reason for stopping."},
                    "status": {
                        "type": "string",
                        "description": "Required only by `action=stop_search`.",
                        "enum": ["success", "failure"],
                    },
                },
            },
        },
    }


def build_search_system_prompt(tools_def: Dict, collapse_text: str) -> str:
    return (
        "You are a browser-search agent that can both browse and save structured knowledge.\n\n"
        "# Tools\n\n"
        "You have access to the following function:\n\n"
        "<tools>\n"
        + json.dumps(tools_def, ensure_ascii=False)
        + "\n</tools>\n\n"
        "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
        "<tool_call>\n"
        "<function=computer_use>\n"
        "<parameter=action>\n"
        "one_allowed_action_name\n"
        "</parameter>\n"
        "<parameter=action_parameter_1>\n"
        "value\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n\n"
        "<IMPORTANT>\n"
        "Reminder:\n"
        "- Function calls MUST follow the specified XML format.\n"
        "- Required parameters MUST be specified.\n"
        "- Action names must be plain enum values with no quotes, such as left_click or save_knowledge.\n"
        f"- The current date is {datetime.today().strftime('%A, %B %d, %Y')}.\n"
        f"- Collapsed screenshots appear as text: {collapse_text}\n"
        "</IMPORTANT>\n\n"
        "# Response format\n\n"
        "For every turn output exactly:\n"
        "1) Action: a short imperative describing the next step.\n"
        "2) A single <tool_call>...</tool_call> block.\n\n"
        "Rules:\n"
        "- Take exactly one action per turn.\n"
        "- Use GUI actions to browse.\n"
        "- Use save_knowledge when the current screenshot or page visibly contains task-relevant knowledge worth preserving.\n"
        "- save_knowledge must be grounded in visible evidence; do not save guesses.\n"
        "- Use stop_search only after enough knowledge has been saved, or when evidence is insufficient and continued browsing is unlikely to help.\n"
        "- Do not output anything after the tool call.\n"
    )


def parse_search_response(
    response: str,
    *,
    coordinate_type: str,
    original_width: int = None,
    original_height: int = None,
    processed_width: int = None,
    processed_height: int = None,
) -> Tuple[str, List[str]]:
    low_level_instruction = ""
    actions: List[str] = []

    if not response or not response.strip():
        return low_level_instruction, actions

    def adjust(x: float, y: float) -> Tuple[int, int]:
        return adjust_coordinates(
            x,
            y,
            coordinate_type=coordinate_type,
            original_width=original_width,
            original_height=original_height,
            processed_width=processed_width,
            processed_height=processed_height,
        )

    def save_special(prefix: str, payload: Dict) -> None:
        actions.append(prefix + json.dumps(payload, ensure_ascii=False))

    for params in iter_tool_call_params(response):
        action = _normalize_search_action(params.get("action"))
        if not action:
            continue

        coordinate = parse_coordinate(params.get("coordinate"))
        if action == "save_knowledge":
            payload = {
                "summary": str(params.get("summary") or ""),
                "steps": params.get("steps") if isinstance(params.get("steps"), list) else params.get("steps", []),
                "source": str(params.get("source") or ""),
                "evidence": str(params.get("evidence") or ""),
                "task_relevance": str(params.get("task_relevance") or ""),
                "uncertainty": str(params.get("uncertainty") or ""),
                "confidence": str(params.get("confidence") or "medium"),
            }
            save_special("SAVE_KNOWLEDGE:", payload)
            continue
        if action == "stop_search":
            payload = {
                "status": str(params.get("status") or "failure"),
                "reason": str(params.get("reason") or ""),
            }
            save_special("STOP_SEARCH:", payload)
            continue
        if action == "left_click":
            if coordinate:
                x, y = adjust(*coordinate)
                actions.append(f"pyautogui.click({x}, {y})")
            else:
                actions.append("pyautogui.click()")
        elif action == "right_click":
            if coordinate:
                x, y = adjust(*coordinate)
                actions.append(f"pyautogui.rightClick({x}, {y})")
            else:
                actions.append("pyautogui.rightClick()")
        elif action == "middle_click":
            if coordinate:
                x, y = adjust(*coordinate)
                actions.append(f"pyautogui.middleClick({x}, {y})")
            else:
                actions.append("pyautogui.middleClick()")
        elif action == "double_click":
            if coordinate:
                x, y = adjust(*coordinate)
                actions.append(f"pyautogui.doubleClick({x}, {y})")
            else:
                actions.append("pyautogui.doubleClick()")
        elif action == "triple_click":
            if coordinate:
                x, y = adjust(*coordinate)
                actions.append(f"pyautogui.tripleClick({x}, {y})")
            else:
                actions.append("pyautogui.tripleClick()")
        elif action == "type":
            text = "" if params.get("text") is None else str(params.get("text", ""))
            actions.append(f"pyautogui.typewrite({py_string(text)})")
        elif action == "key":
            keys = parse_keys(params.get("keys", []), lowercase=True)
            if keys:
                keys_str = ", ".join(py_string(key) for key in keys)
                if len(keys) > 1:
                    actions.append(f"pyautogui.hotkey({keys_str})")
                else:
                    actions.append(f"pyautogui.press({keys_str})")
        elif action == "key_down":
            for key in parse_keys(params.get("keys", []), lowercase=True):
                actions.append(f"pyautogui.keyDown({py_string(key)})")
        elif action == "key_up":
            for key in parse_keys(params.get("keys", []), lowercase=True):
                actions.append(f"pyautogui.keyUp({py_string(key)})")
        elif action in {"scroll", "hscroll"}:
            pixels = int(parse_number(params.get("pixels", 0), default=0))
            clicks = _pixels_to_clicks(pixels)
            if action == "scroll":
                clicks = -clicks
            actions.append(f"pyautogui.{action}({clicks})")
        elif action == "wait":
            actions.append("WAIT")
        elif action == "screenshot":
            actions.append("WAIT")
        elif action == "mouse_move":
            if coordinate:
                x, y = adjust(*coordinate)
                actions.append(f"pyautogui.moveTo({x}, {y})")
            else:
                actions.append("pyautogui.moveTo(0, 0)")
        elif action == "left_click_drag":
            if coordinate:
                x, y = adjust(*coordinate)
                actions.append(f"pyautogui.dragTo({x}, {y}, duration=0.5)")
            else:
                actions.append("pyautogui.dragTo(0, 0)")
        elif action == "left_mouse_down":
            if coordinate:
                x, y = adjust(*coordinate)
                actions.append(f"pyautogui.moveTo({x}, {y})")
            actions.append("pyautogui.mouseDown(button='left')")
        elif action == "left_mouse_up":
            if coordinate:
                x, y = adjust(*coordinate)
                actions.append(f"pyautogui.moveTo({x}, {y})")
            actions.append("pyautogui.mouseUp(button='left')")

    low_level_instruction = extract_action_line(response)
    if not low_level_instruction and actions:
        first = actions[0]
        if first.startswith("SAVE_KNOWLEDGE:"):
            low_level_instruction = "Save visible knowledge"
        elif first.startswith("STOP_SEARCH:"):
            low_level_instruction = "Stop search"
        else:
            low_level_instruction = "Perform action"
    return low_level_instruction, actions


class _GeminiBaseAgent:
    """
    Shared implementation for Gemini computer-use agents.

    Characteristics:
    - Google GenAI SDK.
    - XML tool-call output format.
    - History truncation by `history_n`.
    - Old screenshot folding by `image_max` / `fold_size`.
    """

    COLLAPSED_SCREENSHOT_TEXT = "This screenshot has been collapsed."

    def __init__(
        self,
        platform: str = "ubuntu",
        model: str = "gemini-3.5-flash",
        max_tokens: int = 32768,
        top_p: float = 0.9,
        temperature: float = 0.0,
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
        history_n: int = 100,
        add_thought_prefix: bool = False,
        coordinate_type: str = "relative",
        api_backend: str = "google-genai",
        image_max: int = 20,
        fold_size: int = 10,
        collapse_text: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        media_resolution: Optional[str] = "high",
        thinking_budget: Optional[int] = None,
        include_thoughts: bool = False,
        scaffold_profile: str = "qwen_internal",
        examples: list = None,
        search_mode: bool = False,
    ):
        self.platform = platform
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.history_n = history_n
        self.add_thought_prefix = add_thought_prefix
        self.coordinate_type = coordinate_type
        self.api_backend = api_backend
        self.image_max = int(image_max)
        self.fold_size = int(fold_size)
        self.collapse_text = collapse_text or self.COLLAPSED_SCREENSHOT_TEXT
        self.base_url = base_url
        self.api_key = api_key
        self.media_resolution = media_resolution
        self.thinking_budget = thinking_budget
        self.include_thoughts = include_thoughts
        self.scaffold_profile_name = scaffold_profile
        self.scaffold_profile = get_gemini_scaffold_profile(scaffold_profile)
        self.search_mode = bool(search_mode)
        if isinstance(examples, dict):
            if "description" in examples:
                self.examples = self._get_examples_video_description(examples)
            else:
                self.examples = self._get_examples_traj(examples)
        elif isinstance(examples, str):
            self.examples = self._get_examples(examples)
        else:
            self.examples = []

        if action_space != "pyautogui":
            raise ValueError("GeminiAgent supports only pyautogui action space")
        if observation_type != "screenshot":
            raise ValueError("GeminiAgent supports only screenshot observations")
        if api_backend != "google-genai":
            raise ValueError("GeminiAgent supports only google-genai backend")
        if self.image_max < 1:
            raise ValueError("image_max must be >= 1")
        if self.fold_size < 1:
            raise ValueError("fold_size must be >= 1")

        self.thoughts: List[str] = []
        self.actions: List[str] = []
        self.observations: List[Dict] = []
        self.responses: List[str] = []
        self.screenshots: List[str] = []
        self.folded_prefix_k = 0

    @staticmethod
    def _py_string(text: str) -> str:
        return py_string(text)

    def _build_tools_def(self, processed_width: int, processed_height: int) -> Dict:
        return build_base_tools_def(processed_width, processed_height, self.coordinate_type)

    def _build_system_prompt(self, tools_def: Dict) -> str:
        return build_base_system_prompt(tools_def, self.collapse_text)

    def _response_transform(self, response: str) -> str:
        return response

    def _debug_message_filename(self, step_idx: int) -> str:
        return f"gemini_messages_step_{step_idx}.json"

    def _build_payload(self, messages: List[Dict]) -> Dict:
        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "temperature": self.temperature,
        }

    def _log_prefix(self) -> str:
        return "Gemini"

    def _get_examples(self, example):
        newcontent = []
        if os.path.exists(example):
            tmp_path = create_video_clip(example)
            base64_video = encode_video_base64(tmp_path)
            data_url = f"data:video/mp4;base64,{base64_video}"
            newcontent = [
                {"type": "text", "text": "You are first given the following demonstration video of how to perform a task that you will be performing next. You can use it as a guide or just follow this demonstration."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        elif not example.endswith(".mp4"):
            newcontent = [
                {"type": "text", "text": "You are first given the following detailed description of how to perform a task that you will be performing next. You can use it as a guide or just follow this demonstration."},
                {"type": "text", "text": example},
            ]
        return newcontent

    def _get_examples_video_description(self, example):
        newcontent = [
            {"type": "text", "text": "You are first given the following detailed description of how to perform this task or part of this task. Use it as a guide when it is helpful:"},
            {"type": "text", "text": example["description"]},
        ]
        return newcontent

    def _get_examples_traj(self, examples):
        newcontent = []
        if examples is not None:
            with open(examples["init_screenshot"], "rb") as f:
                init_bytes = f.read()
            init_screenshot = process_image_resize(init_bytes, max_pixels=176400)
            img_url = f"data:image/png;base64,{init_screenshot}"
            newcontent = [
                {"type": "text", "text": "You are first given the following demonstration of how to perform the task. Use this as a guide."},
                {"type": "image_url", "image_url": {"url": img_url}},
            ]
            for example in examples["steps"]:
                with open(example["screenshot"], "rb") as f:
                    this_bytes = f.read()
                this_screenshot = process_image_resize(this_bytes, max_pixels=176400)
                img_url = f"data:image/png;base64,{this_screenshot}"
                newcontent += [
                    {"type": "text", "text": example["response"]},
                    {"type": "image_url", "image_url": {"url": img_url}},
                ]
            newcontent += [
                {"type": "text", "text": "\nThis is the end of the example demonstration.\n\nThe following is the initial screenshot of your task:\n"},
            ]
        return newcontent

    def _parse_response(
        self,
        response: str,
        *,
        original_width: int,
        original_height: int,
        processed_width: int,
        processed_height: int,
    ) -> Tuple[str, List[str]]:
        return parse_base_response(
            response,
            coordinate_type=self.coordinate_type,
            original_width=original_width,
            original_height=original_height,
            processed_width=processed_width,
            processed_height=processed_height,
        )

    def predict(self, instruction: str, obs: Dict) -> Tuple[str, List[str]]:
        screenshot_bytes = obs["screenshot"]

        original_width, original_height = image_size_from_bytes(screenshot_bytes)
        processed_b64 = process_image(screenshot_bytes)
        processed_width, processed_height = image_size_from_base64(processed_b64)

        self.screenshots.append(processed_b64)
        total_steps = len(self.screenshots)
        self.folded_prefix_k = update_folding_state(
            total_steps,
            self.folded_prefix_k,
            self.image_max,
            self.fold_size,
        )

        start_step = max(1, total_steps - self.history_n)
        previous_actions_str = previous_actions_text(self.actions, start_step)

        tools_def = self._build_tools_def(processed_width, processed_height)
        system_prompt = self._build_system_prompt(tools_def)
        if instruction.endswith(".mp3"):
            instruction_prompt = [
                "\nPlease generate the next move according to the UI screenshot, instruction and previous actions.\n\nThe instruction is given in the following audio:\n",
                instruction,
                f"Previous actions:\n{previous_actions_str}",
            ]
        else:
            instruction_prompt = build_instruction_prompt(instruction, previous_actions_str)

        self.observations.append({"screenshot": processed_b64})
        messages = build_messages(
            system_prompt=system_prompt,
            instruction_prompt=instruction_prompt,
            screenshots=self.screenshots,
            responses=self.responses,
            start_step=start_step,
            total_steps=total_steps,
            folded_prefix_k=self.folded_prefix_k,
            collapse_text=self.collapse_text,
            response_transform=self._response_transform,
            demo=self.examples,
        )

        step_idx = total_steps - 1
        dump_debug_messages(messages, self._debug_message_filename(step_idx), logger)

        response = self.call_llm(self._build_payload(messages), self.model)

        if logger:
            logger.info("%s Output: %s", self._log_prefix(), response)
        self.responses.append(response or "")

        low_level_instruction, parsed_actions = self._parse_response(
            response or "",
            original_width=original_width,
            original_height=original_height,
            processed_width=processed_width,
            processed_height=processed_height,
        )

        if logger:
            logger.info("Low level instruction: %s", low_level_instruction)
            logger.info("Pyautogui code: %s", parsed_actions)

        self.actions.append(low_level_instruction)
        return response or "", parsed_actions

    def parse_response(
        self,
        response: str,
        original_width: int = None,
        original_height: int = None,
        processed_width: int = None,
        processed_height: int = None,
    ) -> Tuple[str, List[str]]:
        return parse_base_response(
            response,
            coordinate_type=self.coordinate_type,
            original_width=original_width,
            original_height=original_height,
            processed_width=processed_width,
            processed_height=processed_height,
        )

    def call_llm(self, payload: Dict, model: str) -> str:
        return call_gemini(
            payload,
            model,
            base_url=self.base_url,
            api_key=self.api_key,
            default_max_tokens=self.max_tokens,
            default_temperature=self.temperature,
            default_top_p=self.top_p,
            media_resolution=self.media_resolution,
            thinking_budget=self.thinking_budget,
            include_thoughts=self.include_thoughts,
            logger=logger,
        )

    def reset(self, _logger=None, *args, **kwargs):
        global logger
        logger = _logger if _logger is not None else logging.getLogger("desktopenv.gemini_agent")
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.responses = []
        self.screenshots = []
        self.folded_prefix_k = 0


class GeminiAgent(_GeminiBaseAgent):
    COLLAPSED_SCREENSHOT_TEXT = "This screenshot has been collapsed."

    def __init__(
        self,
        *args,
        observation_type: str = "screenshot",
        search_mode: bool = False,
        **kwargs,
    ):
        super().__init__(*args, observation_type="screenshot", search_mode=search_mode, **kwargs)
        if observation_type != "screenshot":
            raise ValueError("GeminiAgent supports only screenshot observations")
        self.observation_type = observation_type
        self.search_mode = bool(search_mode)

    def _build_tools_def(self, processed_width: int, processed_height: int) -> Dict:
        if self.search_mode:
            return build_search_tools_def(processed_width, processed_height, self.coordinate_type)
        return self.scaffold_profile.build_tools_def(
            processed_width,
            processed_height,
            self.coordinate_type,
        )

    def _build_system_prompt(self, tools_def: Dict) -> str:
        if self.search_mode:
            return build_search_system_prompt(tools_def, self.collapse_text)
        return self.scaffold_profile.build_system_prompt(tools_def, self.collapse_text)

    def _response_transform(self, response: str) -> str:
        if self.search_mode:
            return ensure_empty_think_prefix(response)
        return self.scaffold_profile.response_transform(response)

    def _debug_message_filename(self, step_idx: int) -> str:
        return f"gemini_messages_step_{step_idx}.json"

    def _log_prefix(self) -> str:
        return "Gemini"

    def _parse_response(
        self,
        response: str,
        *,
        original_width: int,
        original_height: int,
        processed_width: int,
        processed_height: int,
    ) -> Tuple[str, List[str]]:
        if self.search_mode:
            return parse_search_response(
                response,
                coordinate_type=self.coordinate_type,
                original_width=original_width,
                original_height=original_height,
                processed_width=processed_width,
                processed_height=processed_height,
            )
        return self.scaffold_profile.parse_response(
            response,
            coordinate_type=self.coordinate_type,
            original_width=original_width,
            original_height=original_height,
            processed_width=processed_width,
            processed_height=processed_height,
        )

    def parse_response(
        self,
        response: str,
        original_width: int = None,
        original_height: int = None,
        processed_width: int = None,
        processed_height: int = None,
    ) -> Tuple[str, List[str]]:
        if self.search_mode:
            return parse_search_response(
                response,
                coordinate_type=self.coordinate_type,
                original_width=original_width,
                original_height=original_height,
                processed_width=processed_width,
                processed_height=processed_height,
            )
        return self.scaffold_profile.parse_response(
            response,
            coordinate_type=self.coordinate_type,
            original_width=original_width,
            original_height=original_height,
            processed_width=processed_width,
            processed_height=processed_height,
        )