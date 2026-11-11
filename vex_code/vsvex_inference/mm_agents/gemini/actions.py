import json
from typing import Dict, List, Tuple

from .constants import SET_CELL_VALUES_CMD
from .images import adjust_coordinates
from .parser import (
    extract_action_line,
    iter_tool_call_params,
    looks_infeasible_response,
    parse_coordinate,
    parse_keys,
    parse_number,
)

SCROLL_PIXELS_PER_CLICK = 120.0
SCROLL_MAX_CLICKS = 8


def _pixels_to_clicks(pixels: int) -> int:
    if pixels == 0:
        return 0
    magnitude = min(
        SCROLL_MAX_CLICKS,
        max(1, int(abs(pixels) / SCROLL_PIXELS_PER_CLICK + 0.5)),
    )
    return magnitude if pixels > 0 else -magnitude


def py_string(text: str) -> str:
    return json.dumps("" if text is None else str(text), ensure_ascii=False)


def _termination_code(status: object) -> str:
    normalized = str(status or "success").strip().lower()
    return "FAIL" if normalized in {"fail", "failed", "failure", "error", "infeasible"} else "DONE"


def _coord_adjuster(
    *,
    coordinate_type: str,
    original_width: int = None,
    original_height: int = None,
    processed_width: int = None,
    processed_height: int = None,
):
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

    return adjust


def _parse_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _join_py_commands(commands: List[str]) -> str:
    return "; ".join(command for command in commands if command)


def _append_text_typing(commands: List[str], text: str) -> None:
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    if "\n" not in normalized_text:
        commands.append(f"pyautogui.typewrite({py_string(normalized_text)})")
        return

    chunks = normalized_text.split("\n")
    for idx, chunk in enumerate(chunks):
        if chunk:
            commands.append(f"pyautogui.typewrite({py_string(chunk)})")
        if idx < len(chunks) - 1:
            commands.append(f"pyautogui.press({py_string('enter')})")


def _keys_code(keys: List[str], *, hotkey: bool = True) -> str:
    keys_str = ", ".join(py_string(key) for key in keys)
    if hotkey and len(keys) > 1:
        return f"pyautogui.hotkey({keys_str})"
    return f"pyautogui.press({keys_str})"


def parse_base_response(
    response: str,
    *,
    coordinate_type: str,
    original_width: int = None,
    original_height: int = None,
    processed_width: int = None,
    processed_height: int = None,
) -> Tuple[str, List[str]]:
    low_level_instruction = ""
    pyautogui_code: List[str] = []

    if not response or not response.strip():
        return low_level_instruction, pyautogui_code

    adjust = _coord_adjuster(
        coordinate_type=coordinate_type,
        original_width=original_width,
        original_height=original_height,
        processed_width=processed_width,
        processed_height=processed_height,
    )

    def process_tool_call_params(params: Dict) -> None:
        action = params.get("action")
        if not action:
            return

        coordinate = parse_coordinate(params.get("coordinate"))
        text = params.get("text")

        def press_modifier_keys() -> None:
            if text:
                for key in str(text).split("+"):
                    key = key.strip().lower()
                    if key:
                        pyautogui_code.append(f"pyautogui.keyDown({py_string(key)})")

        def release_modifier_keys() -> None:
            if text:
                keys = [key.strip().lower() for key in str(text).split("+") if key.strip()]
                for key in reversed(keys):
                    pyautogui_code.append(f"pyautogui.keyUp({py_string(key)})")

        if action == "left_click":
            press_modifier_keys()
            if coordinate:
                x, y = adjust(*coordinate)
                pyautogui_code.append(f"pyautogui.click({x}, {y})")
            else:
                pyautogui_code.append("pyautogui.click()")
            release_modifier_keys()
        elif action == "right_click":
            press_modifier_keys()
            if coordinate:
                x, y = adjust(*coordinate)
                pyautogui_code.append(f"pyautogui.rightClick({x}, {y})")
            else:
                pyautogui_code.append("pyautogui.rightClick()")
            release_modifier_keys()
        elif action == "middle_click":
            press_modifier_keys()
            if coordinate:
                x, y = adjust(*coordinate)
                pyautogui_code.append(f"pyautogui.middleClick({x}, {y})")
            else:
                pyautogui_code.append("pyautogui.middleClick()")
            release_modifier_keys()
        elif action == "double_click":
            press_modifier_keys()
            if coordinate:
                x, y = adjust(*coordinate)
                pyautogui_code.append(f"pyautogui.doubleClick({x}, {y})")
            else:
                pyautogui_code.append("pyautogui.doubleClick()")
            release_modifier_keys()
        elif action == "triple_click":
            press_modifier_keys()
            if coordinate:
                x, y = adjust(*coordinate)
                pyautogui_code.append(f"pyautogui.doubleClick({x}, {y})")
            else:
                pyautogui_code.append("pyautogui.doubleClick()")
            release_modifier_keys()
        elif action == "type":
            text = params.get("text", "")
            pyautogui_code.append(f"pyautogui.typewrite({py_string(text)})")
        elif action == "key":
            keys = parse_keys(params.get("keys", []))
            keys_str = ", ".join(py_string(key) for key in keys)
            if len(keys) > 1:
                pyautogui_code.append(f"pyautogui.hotkey({keys_str})")
            else:
                pyautogui_code.append(f"pyautogui.press({keys_str})")
        # elif action in {"scroll", "hscroll"}:
        #     press_modifier_keys()
        #     pixels = params.get("pixels", 0)
        #     try:
        #         pixels = int(float(pixels))
        #     except Exception:
        #         pixels = 0
        #     pyautogui_code.append(f"pyautogui.scroll({pixels})")
        #     release_modifier_keys()
        elif action in {"scroll", "hscroll"}:
            press_modifier_keys()
            pixels = params.get("pixels", 0)
            try:
                pixels = int(float(pixels))
            except (TypeError, ValueError, OverflowError):
                pixels = 0

            clicks = _pixels_to_clicks(pixels)
            if action == "scroll":
                clicks = -clicks

            pyautogui_code.append(f"pyautogui.{action}({clicks})")
            release_modifier_keys()
        elif action == "wait":
            pyautogui_code.append("WAIT")
        elif action == "terminate":
            pyautogui_code.append(_termination_code(params.get("status", "success")))
        elif action == "answer":
            pyautogui_code.append("DONE")
        elif action == "mouse_move":
            if coordinate:
                x, y = adjust(*coordinate)
                pyautogui_code.append(f"pyautogui.moveTo({x}, {y})")
            else:
                pyautogui_code.append("pyautogui.moveTo(0, 0)")
        elif action == "left_click_drag":
            if coordinate:
                x, y = adjust(*coordinate)
                duration = 0.5
                if "duration" in params:
                    try:
                        duration = float(params["duration"])
                    except Exception:
                        duration = 0.5
                pyautogui_code.append(f"pyautogui.dragTo({x}, {y}, duration={duration})")
            else:
                pyautogui_code.append("pyautogui.dragTo(0, 0)")

    low_level_instruction = extract_action_line(response, preserve_base_split_bug=True)

    for params in iter_tool_call_params(response):
        process_tool_call_params(params)

    if not low_level_instruction and pyautogui_code:
        low_level_instruction = _instruction_from_first_code(pyautogui_code[0])

    return low_level_instruction, pyautogui_code


def parse_internal_response(
    response: str,
    *,
    coordinate_type: str,
    original_width: int = None,
    original_height: int = None,
    processed_width: int = None,
    processed_height: int = None,
) -> Tuple[str, List[str]]:
    low_level_instruction = ""
    pyautogui_code: List[str] = []

    if not response or not response.strip():
        return low_level_instruction, pyautogui_code

    adjust = _coord_adjuster(
        coordinate_type=coordinate_type,
        original_width=original_width,
        original_height=original_height,
        processed_width=processed_width,
        processed_height=processed_height,
    )
    infeasible_response = looks_infeasible_response(response)

    def process_tool_call_params(params: Dict) -> None:
        action = params.get("action")
        if not action:
            return

        coordinate = parse_coordinate(params.get("coordinate"))

        if action == "left_click":
            if coordinate:
                x, y = adjust(*coordinate)
                pyautogui_code.append(f"pyautogui.click({x}, {y})")
            else:
                pyautogui_code.append("pyautogui.click()")
        elif action == "right_click":
            if coordinate:
                x, y = adjust(*coordinate)
                pyautogui_code.append(f"pyautogui.rightClick({x}, {y})")
            else:
                pyautogui_code.append("pyautogui.rightClick()")
        elif action == "middle_click":
            if coordinate:
                x, y = adjust(*coordinate)
                pyautogui_code.append(f"pyautogui.middleClick({x}, {y})")
            else:
                pyautogui_code.append("pyautogui.middleClick()")
        elif action == "double_click":
            if coordinate:
                x, y = adjust(*coordinate)
                pyautogui_code.append(f"pyautogui.doubleClick({x}, {y})")
            else:
                pyautogui_code.append("pyautogui.doubleClick()")
        elif action == "triple_click":
            if coordinate:
                x, y = adjust(*coordinate)
                pyautogui_code.append(f"pyautogui.tripleClick({x}, {y})")
            else:
                pyautogui_code.append("pyautogui.tripleClick()")
        elif action == "type":
            text = "" if params.get("text") is None else str(params.get("text", ""))
            normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
            if "\n" not in normalized_text:
                pyautogui_code.append(f"pyautogui.typewrite({py_string(normalized_text)})")
            else:
                chunks = normalized_text.split("\n")
                for idx, chunk in enumerate(chunks):
                    if chunk:
                        pyautogui_code.append(f"pyautogui.typewrite({py_string(chunk)})")
                    if idx < len(chunks) - 1:
                        pyautogui_code.append(f"pyautogui.press({py_string('enter')})")
        elif action == "key":
            keys = parse_keys(params.get("keys", []), lowercase=True)
            if not keys:
                return
            keys_str = ", ".join(py_string(key) for key in keys)
            if len(keys) > 1:
                pyautogui_code.append(f"pyautogui.hotkey({keys_str})")
            else:
                pyautogui_code.append(f"pyautogui.press({keys_str})")
        elif action == "key_down":
            for key in parse_keys(params.get("keys", []), lowercase=True):
                pyautogui_code.append(f"pyautogui.keyDown({py_string(key)})")
        elif action == "key_up":
            for key in parse_keys(params.get("keys", []), lowercase=True):
                pyautogui_code.append(f"pyautogui.keyUp({py_string(key)})")
        # elif action == "scroll":
        #     pixels = int(parse_number(params.get("pixels", 0), default=0))
        #     pyautogui_code.append(f"pyautogui.scroll({pixels})")
        # elif action == "hscroll":
        #     pixels = int(parse_number(params.get("pixels", 0), default=0))
        #     pyautogui_code.append(f"pyautogui.hscroll({pixels})")
        elif action in {"scroll", "hscroll"}:
            pixels = int(parse_number(params.get("pixels", 0), default=0))
            clicks = _pixels_to_clicks(pixels)
            if action == "scroll":
                clicks = -clicks
            pyautogui_code.append(f"pyautogui.{action}({clicks})")
        elif action == "wait":
            pyautogui_code.append("WAIT")
        elif action == "terminate":
            pyautogui_code.append(_termination_code(params.get("status", "success")))
        elif action == "call_user":
            pyautogui_code.append("FAIL" if infeasible_response else "DONE")
        elif action == "screenshot":
            pyautogui_code.append("WAIT")
        elif action == "mouse_move":
            if coordinate:
                x, y = adjust(*coordinate)
                pyautogui_code.append(f"pyautogui.moveTo({x}, {y})")
            else:
                pyautogui_code.append("pyautogui.moveTo(0, 0)")
        elif action == "left_click_drag":
            if coordinate:
                x, y = adjust(*coordinate)
                duration = parse_number(params.get("duration", 0.5), default=0.5)
                pyautogui_code.append(f"pyautogui.dragTo({x}, {y}, duration={duration})")
            else:
                pyautogui_code.append("pyautogui.dragTo(0, 0)")
        elif action == "left_mouse_down":
            if coordinate:
                x, y = adjust(*coordinate)
                pyautogui_code.append(f"pyautogui.moveTo({x}, {y})")
            pyautogui_code.append("pyautogui.mouseDown(button='left')")
        elif action == "left_mouse_up":
            if coordinate:
                x, y = adjust(*coordinate)
                pyautogui_code.append(f"pyautogui.moveTo({x}, {y})")
            pyautogui_code.append("pyautogui.mouseUp(button='left')")

    low_level_instruction = extract_action_line(response)

    for params in iter_tool_call_params(response):
        process_tool_call_params(params)

    if not pyautogui_code:
        pyautogui_code.append("FAIL" if infeasible_response else "DONE")

    if not low_level_instruction and pyautogui_code:
        first_code = pyautogui_code[0]
        if first_code == "FAIL":
            low_level_instruction = "Need user input"
        else:
            low_level_instruction = _instruction_from_first_code(first_code)

    return low_level_instruction, pyautogui_code


def parse_hybrid_macro_response(
    response: str,
    *,
    coordinate_type: str,
    original_width: int = None,
    original_height: int = None,
    processed_width: int = None,
    processed_height: int = None,
) -> Tuple[str, List[str]]:
    low_level_instruction = ""
    pyautogui_code: List[str] = []

    if not response or not response.strip():
        return low_level_instruction, pyautogui_code

    adjust = _coord_adjuster(
        coordinate_type=coordinate_type,
        original_width=original_width,
        original_height=original_height,
        processed_width=processed_width,
        processed_height=processed_height,
    )
    infeasible_response = looks_infeasible_response(response)

    def adjusted_coordinate(params: Dict, key: str = "coordinate"):
        coordinate = parse_coordinate(params.get(key))
        if not coordinate:
            return None
        return adjust(*coordinate)

    def process_tool_call_params(params: Dict) -> None:
        action = params.get("action")
        if not action:
            return

        if action == "hotkey":
            keys = parse_keys(params.get("keys", []), lowercase=True)
            if keys:
                pyautogui_code.append(_keys_code(keys))
        elif action == "key_down":
            commands = [
                f"pyautogui.keyDown({py_string(key)})"
                for key in parse_keys(params.get("keys", []), lowercase=True)
            ]
            if commands:
                pyautogui_code.append(_join_py_commands(commands))
        elif action == "key_up":
            commands = [
                f"pyautogui.keyUp({py_string(key)})"
                for key in parse_keys(params.get("keys", []), lowercase=True)
            ]
            if commands:
                pyautogui_code.append(_join_py_commands(commands))
        elif action == "mouse_move":
            xy = adjusted_coordinate(params)
            if xy:
                x, y = xy
                pyautogui_code.append(f"pyautogui.moveTo({x}, {y})")
            else:
                pyautogui_code.append("pyautogui.moveTo(0, 0)")
        elif action == "left_mouse_down":
            commands = []
            xy = adjusted_coordinate(params)
            if xy:
                x, y = xy
                commands.append(f"pyautogui.moveTo({x}, {y})")
            commands.append("pyautogui.mouseDown(button='left')")
            pyautogui_code.append(_join_py_commands(commands))
        elif action == "left_mouse_up":
            commands = []
            xy = adjusted_coordinate(params)
            if xy:
                x, y = xy
                commands.append(f"pyautogui.moveTo({x}, {y})")
            commands.append("pyautogui.mouseUp(button='left')")
            pyautogui_code.append(_join_py_commands(commands))
        elif action == "click":
            xy = adjusted_coordinate(params)
            if not xy:
                return
            x, y = xy
            commands = []
            hold_keys = parse_keys(params.get("hold_keys", []), lowercase=True)
            for key in hold_keys:
                commands.append(f"pyautogui.keyDown({py_string(key)})")
            commands.append(f"pyautogui.moveTo({x}, {y})")

            button_type = str(params.get("button_type") or "left").strip().lower()
            if button_type not in {"left", "middle", "right"}:
                button_type = "left"
            num_clicks = int(parse_number(params.get("num_clicks", 1), default=1))
            duration = parse_number(params.get("duration", 0), default=0)

            if duration > 0 and num_clicks <= 1:
                commands.append(f"pyautogui.mouseDown(button={py_string(button_type)})")
                commands.append(f"time.sleep({duration})")
                commands.append(f"pyautogui.mouseUp(button={py_string(button_type)})")
            elif button_type == "left" and num_clicks == 2:
                commands.append(f"pyautogui.doubleClick({x}, {y})")
            elif button_type == "left" and num_clicks == 3:
                commands.append(f"pyautogui.tripleClick({x}, {y})")
            elif button_type == "right" and num_clicks == 1:
                commands.append(f"pyautogui.rightClick({x}, {y})")
            elif button_type == "middle" and num_clicks == 1:
                commands.append(f"pyautogui.middleClick({x}, {y})")
            else:
                commands.append(
                    f"pyautogui.click({x}, {y}, button={py_string(button_type)}, clicks={max(num_clicks, 1)}, interval=0.25)"
                )

            for key in reversed(hold_keys):
                commands.append(f"pyautogui.keyUp({py_string(key)})")
            pyautogui_code.append(_join_py_commands(commands))
        elif action == "type":
            commands = []
            xy = adjusted_coordinate(params)
            if xy:
                x, y = xy
                commands.append(f"pyautogui.click({x}, {y})")
            if _parse_bool(params.get("overwrite"), False):
                commands.append("pyautogui.hotkey('ctrl', 'a')")
                commands.append("pyautogui.press('backspace')")
            text = "" if params.get("text") is None else str(params.get("text", ""))
            _append_text_typing(commands, text)
            if _parse_bool(params.get("enter"), False):
                commands.append("pyautogui.press('enter')")
            if commands:
                pyautogui_code.append(_join_py_commands(commands))
        elif action == "scroll":
            commands = []
            xy = adjusted_coordinate(params)
            if xy:
                x, y = xy
                commands.append(f"pyautogui.moveTo({x}, {y})")
            clicks = int(parse_number(params.get("clicks", 0), default=0))
            scroll_amount = -clicks
            if _parse_bool(params.get("shift"), False):
                commands.append("pyautogui.keyDown('shift')")
                commands.append(f"pyautogui.scroll({scroll_amount})")
                commands.append("pyautogui.keyUp('shift')")
            else:
                commands.append(f"pyautogui.scroll({scroll_amount})")
            pyautogui_code.append(_join_py_commands(commands))
        elif action == "wait":
            pyautogui_code.append("WAIT")
        elif action == "terminate":
            pyautogui_code.append(_termination_code(params.get("status", "success")))
        elif action == "call_user":
            pyautogui_code.append("FAIL" if infeasible_response else "DONE")
        elif action == "screenshot":
            pyautogui_code.append("WAIT")
        elif action == "drag_and_drop":
            start_xy = adjusted_coordinate(params, "start_coordinate")
            end_xy = adjusted_coordinate(params, "end_coordinate")
            if not (start_xy and end_xy):
                return
            x1, y1 = start_xy
            x2, y2 = end_xy
            commands = [f"pyautogui.moveTo({x1}, {y1})"]
            hold_keys = parse_keys(params.get("hold_keys", []), lowercase=True)
            for key in hold_keys:
                commands.append(f"pyautogui.keyDown({py_string(key)})")
            commands.append(f"pyautogui.dragTo({x2}, {y2}, duration=1.0)")
            commands.append("pyautogui.mouseUp(button='left')")
            for key in reversed(hold_keys):
                commands.append(f"pyautogui.keyUp({py_string(key)})")
            pyautogui_code.append(_join_py_commands(commands))
        elif action == "highlight_text_span":
            start_xy = adjusted_coordinate(params, "start_coordinate")
            end_xy = adjusted_coordinate(params, "end_coordinate")
            if not (start_xy and end_xy):
                return
            x1, y1 = start_xy
            x2, y2 = end_xy
            pyautogui_code.append(
                _join_py_commands(
                    [
                        f"pyautogui.moveTo({x1}, {y1})",
                        f"pyautogui.dragTo({x2}, {y2}, duration=1.0)",
                        "pyautogui.mouseUp(button='left')",
                    ]
                )
            )
        elif action == "hold_and_press":
            commands = []
            hold_keys = parse_keys(params.get("hold_keys", []), lowercase=True)
            press_keys = parse_keys(params.get("press_keys", []), lowercase=True)
            for key in hold_keys:
                commands.append(f"pyautogui.keyDown({py_string(key)})")
            for key in press_keys:
                commands.append(f"pyautogui.press({py_string(key)})")
            for key in reversed(hold_keys):
                commands.append(f"pyautogui.keyUp({py_string(key)})")
            if commands:
                pyautogui_code.append(_join_py_commands(commands))
        elif action == "open":
            app_or_filename = "" if params.get("app_or_filename") is None else str(params.get("app_or_filename", ""))
            pyautogui_code.append(
                _join_py_commands(
                    [
                        "pyautogui.hotkey('win')",
                        "time.sleep(1)",
                        f"pyautogui.typewrite({py_string(app_or_filename)})",
                        "time.sleep(1)",
                        "pyautogui.press('enter')",
                    ]
                )
            )
        elif action == "switch_applications":
            app_code = "" if params.get("app_code") is None else str(params.get("app_code", ""))
            command = "wmctrl -a " + py_string(app_code)
            pyautogui_code.append(f"import os; os.system({py_string(command)})")
        elif action == "set_cell_values":
            cell_values = params.get("cell_values", {})
            if isinstance(cell_values, str):
                try:
                    cell_values = json.loads(cell_values)
                except Exception:
                    cell_values = {}
            if not isinstance(cell_values, dict):
                cell_values = {}
            app_name = "Untitled 1" if params.get("app_name") is None else str(params.get("app_name", "Untitled 1"))
            sheet_name = "Sheet1" if params.get("sheet_name") is None else str(params.get("sheet_name", "Sheet1"))
            pyautogui_code.append(
                SET_CELL_VALUES_CMD.format(
                    cell_values=repr(cell_values),
                    app_name=py_string(app_name),
                    sheet_name=py_string(sheet_name),
                )
            )

    low_level_instruction = extract_action_line(response)

    for params in iter_tool_call_params(response):
        process_tool_call_params(params)

    if not pyautogui_code:
        pyautogui_code.append("FAIL" if infeasible_response else "FAIL")

    if not low_level_instruction and pyautogui_code:
        first_code = pyautogui_code[0]
        if first_code == "FAIL":
            low_level_instruction = "Invalid or infeasible action"
        else:
            low_level_instruction = _instruction_from_first_code(first_code)

    return low_level_instruction, pyautogui_code


def _instruction_from_first_code(first_code: str) -> str:
    if first_code == "DONE":
        return "Task completed"
    if first_code == "WAIT":
        return "Waiting"
    if "." in first_code:
        return f"Performing {first_code.split('.', 1)[1].split('(', 1)[0]} action"
    return "Performing action"
