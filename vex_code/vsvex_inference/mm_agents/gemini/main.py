import logging
from typing import Dict, List, Optional, Tuple
import sys, os

from .actions import parse_base_response, py_string
from .client import call_gemini
from .history import (
    build_messages,
    dump_debug_messages,
    previous_actions_text,
    update_folding_state,
)
from .images import image_size_from_base64, image_size_from_bytes, process_image, process_image_resize
from .images import create_video_clip, encode_video_base64
from .prompts import (
    build_base_system_prompt,
    build_base_tools_def,
    build_instruction_prompt,
)
from .scaffolds import get_gemini_scaffold_profile


logger = None


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
            with open(examples["init_screenshot"], 'rb') as f:
                init_bytes = f.read()
            init_screenshot = process_image_resize(init_bytes, max_pixels=176400)
            img_url = f"data:image/png;base64,{init_screenshot}"
            newcontent = [
                {"type": "text", "text": "You are first given the following demonstration of how to perform the task. Use this as a guide."},
                {"type": "image_url", "image_url": {"url": img_url}},
            ]
            for example in examples["steps"]:
                with open(example["screenshot"], 'rb') as f:
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

        original_width, original_height = image_size_from_bytes(screenshot_bytes) # (1920,1080)
        processed_b64 = process_image(screenshot_bytes)
        processed_width, processed_height = image_size_from_base64(processed_b64) # (1920,1080)

        self.screenshots.append(processed_b64)
        total_steps = len(self.screenshots) # starts from 1
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
                f"Previous actions:\n{previous_actions_str}"
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

        low_level_instruction, pyautogui_code = self._parse_response(
            response or "",
            original_width=original_width,
            original_height=original_height,
            processed_width=processed_width,
            processed_height=processed_height,
        )

        if logger:
            logger.info("Low level instruction: %s", low_level_instruction)
            logger.info("Pyautogui code: %s", pyautogui_code)

        self.actions.append(low_level_instruction)
        return response or "", pyautogui_code

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
        **kwargs,
    ):
        super().__init__(*args, observation_type="screenshot", **kwargs)
        if observation_type != "screenshot":
            raise ValueError("GeminiAgent supports only screenshot observations")
        self.observation_type = observation_type

    def _build_tools_def(self, processed_width: int, processed_height: int) -> Dict:
        return self.scaffold_profile.build_tools_def(
            processed_width,
            processed_height,
            self.coordinate_type,
        )

    def _build_system_prompt(self, tools_def: Dict) -> str:
        return self.scaffold_profile.build_system_prompt(tools_def, self.collapse_text)

    def _response_transform(self, response: str) -> str:
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
        return self.scaffold_profile.parse_response(
            response,
            coordinate_type=self.coordinate_type,
            original_width=original_width,
            original_height=original_height,
            processed_width=processed_width,
            processed_height=processed_height,
        )
