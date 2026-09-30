import json
from datetime import datetime
from typing import Dict


BASE_ACTION_DESCRIPTION_PROMPT = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys (e.g., "ctrl", "shift", "ctrl+shift") that will be held during the click.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) coordinate.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys that will be held during the click.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys that will be held during the click.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys that will be held during the click.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen (simulated as double-click since it's the closest action). Optional `text` parameter can specify modifier keys that will be held during the click.
* `scroll`: Performs a scroll of the mouse scroll wheel. Optional `text` parameter can specify a modifier key (e.g., "shift", "ctrl") that will be held during scrolling.
* `hscroll`: Performs a horizontal scroll (mapped to regular scroll). Optional `text` parameter can specify a modifier key that will be held during scrolling.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Answer a question."""


INTERNAL_ACTION_DESCRIPTION_PROMPT = """
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
* `terminate`: Terminate the current task and report its completion status.
* `call_user`: Ask user for information or confirmation.
""".strip()


HYBRID_MACRO_ACTION_DESCRIPTION_PROMPT = """
* `key_down`: Press and hold a single key without releasing it.
* `key_up`: Release a previously held single key.
* `left_mouse_down`: Press and hold the left mouse button.
* `left_mouse_up`: Release the left mouse button.
* `hotkey`: Press a hotkey combination.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `click`: Move the cursor to a specified (x, y) pixel coordinate on the screen and click. Optional `button_type` can be "left", "middle", or "right". Optional `num_clicks` controls single/double/triple clicks. Optional `duration` controls press-and-hold. Optional `hold_keys` can specify modifier keys to hold during the click.
* `type`: Type a string of text on the keyboard. Optional `coordinate` can first move to a visible input field. Optional `overwrite` clears existing text before typing. Optional `enter` presses Enter after typing.
* `scroll`: Scroll the mouse wheel. Optional `coordinate` can first move the cursor to a visible area. Optional `shift` can be used while scrolling.
* `screenshot`: Capture a new screenshot of the current screen.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `call_user`: Ask user for information or confirmation.
* `drag_and_drop`: Drag from a visible start coordinate to a visible end coordinate.
* `highlight_text_span`: Highlight visible text by dragging from start coordinate to end coordinate.
* `hold_and_press`: Hold a list of keys and press a list of keys.
* `open`: Open any application or file with name app_or_filename.
* `switch_applications`: Switch to a different application that is already open.
* `set_cell_values`: Set values in spreadsheet cells.
""".strip()


def build_description_prompt(processed_width: int, processed_height: int, coordinate_type: str) -> str:
    resolution = (
        f"* The screen's resolution is {processed_width}x{processed_height}."
        if coordinate_type == "absolute"
        else "* The screen's resolution is 1000x1000."
    )
    return "\n".join(
        [
            "Use a mouse and keyboard to interact with a computer, and take screenshots.",
            "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
            "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.",
            resolution,
            "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
            "* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
        ]
    )


def build_base_tools_def(processed_width: int, processed_height: int, coordinate_type: str) -> Dict:
    return {
        "type": "function",
        "function": {
            "name": "computer_use",
            "description": build_description_prompt(processed_width, processed_height, coordinate_type),
            "parameters": {
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {
                        "type": "string",
                        "description": BASE_ACTION_DESCRIPTION_PROMPT,
                        "enum": [
                            "key",
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
                            "wait",
                            "terminate",
                            "answer",
                        ],
                    },
                    "keys": {"type": "array", "description": "Required only by `action=key`."},
                    "text": {
                        "type": "string",
                        "description": "Required by `action=type` and `action=answer`. Optional for click actions (left_click, right_click, middle_click, double_click, triple_click) to specify modifier keys (e.g., 'ctrl', 'shift', 'ctrl+shift'). Optional for scroll actions (scroll, hscroll) to specify a modifier key (e.g., 'shift', 'ctrl') to hold during scrolling.",
                    },
                    "coordinate": {"type": "array", "description": "(x, y) coordinates."},
                    "pixels": {"type": "number", "description": "Scroll amount."},
                    "time": {"type": "number", "description": "Seconds to wait."},
                    "status": {
                        "type": "string",
                        "description": "Task status for terminate.",
                        "enum": ["success", "failure"],
                    },
                },
            },
        },
    }


def build_internal_tools_def(processed_width: int, processed_height: int, coordinate_type: str) -> Dict:
    return {
        "type": "function",
        "function": {
            "name": "computer_use",
            "description": build_description_prompt(processed_width, processed_height, coordinate_type),
            "parameters": {
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {
                        "type": "string",
                        "description": INTERNAL_ACTION_DESCRIPTION_PROMPT,
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
                            "terminate",
                            "call_user",
                        ],
                    },
                    "keys": {
                        "type": "array",
                        "description": "Required only by `action=key`, `action=key_down`, or `action=key_up`.",
                    },
                    "text": {
                        "type": "string",
                        "description": "Required only by `action=type` and `action=call_user`.",
                    },
                    "coordinate": {
                        "type": "array",
                        "description": "(x, y) coordinates. Required only by `action=mouse_move` and `action=left_click_drag`, optional for `action=left_mouse_down` and `action=left_mouse_up`.",
                    },
                    "pixels": {
                        "type": "number",
                        "description": "Scroll amount. Required only by `action=scroll` or `action=hscroll`.",
                    },
                    "time": {
                        "type": "number",
                        "description": "Seconds to wait. Required only by `action=wait`.",
                    },
                    "status": {
                        "type": "string",
                        "description": "Task status for terminate.",
                        "enum": ["success", "failure"],
                    },
                },
            },
        },
    }


def build_hybrid_macro_tools_def(processed_width: int, processed_height: int, coordinate_type: str) -> Dict:
    return {
        "type": "function",
        "function": {
            "name": "computer_use",
            "description": build_description_prompt(processed_width, processed_height, coordinate_type),
            "parameters": {
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {
                        "type": "string",
                        "description": (
                            "Required. This MUST be the first parameter inside <function=computer_use>. "
                            "Use one of the enum values below as the action name. Do not use action names "
                            "as function names; the function name MUST always be computer_use.\n"
                            + HYBRID_MACRO_ACTION_DESCRIPTION_PROMPT
                        ),
                        "enum": [
                            "key_down",
                            "key_up",
                            "left_mouse_down",
                            "left_mouse_up",
                            "hotkey",
                            "mouse_move",
                            "click",
                            "type",
                            "scroll",
                            "screenshot",
                            "wait",
                            "terminate",
                            "call_user",
                            "drag_and_drop",
                            "highlight_text_span",
                            "hold_and_press",
                            "open",
                            "switch_applications",
                            "set_cell_values",
                        ],
                    },
                    "keys": {
                        "type": "array",
                        "description": "Required only by `action=hotkey`, `action=key_down`, or `action=key_up`.",
                    },
                    "text": {
                        "type": "string",
                        "description": "Required only by `action=type` and `action=call_user`.",
                    },
                    "coordinate": {
                        "type": "array",
                        "description": "(x, y) coordinates. Required only by `action=mouse_move` and `action=click`. Optional for `action=left_mouse_down`, `action=left_mouse_up`, `action=type`, and `action=scroll`.",
                    },
                    "start_coordinate": {
                        "type": "array",
                        "description": "Required only by `action=drag_and_drop` and `action=highlight_text_span`.",
                    },
                    "end_coordinate": {
                        "type": "array",
                        "description": "Required only by `action=drag_and_drop` and `action=highlight_text_span`.",
                    },
                    "hold_keys": {
                        "type": "array",
                        "description": "Optional for `action=click` and `action=drag_and_drop`.",
                    },
                    "button_type": {
                        "type": "string",
                        "description": "Optional for `action=click`. One of `left`, `middle`, or `right`. Defaults to `left`.",
                        "enum": ["left", "middle", "right"],
                    },
                    "num_clicks": {
                        "type": "number",
                        "description": "Optional for `action=click`. Defaults to 1.",
                    },
                    "duration": {
                        "type": "number",
                        "description": "Optional for `action=click`. Use only for press-and-hold. Defaults to 0.",
                    },
                    "clicks": {
                        "type": "number",
                        "description": "Required only by `action=scroll`. Positive numbers scroll down; negative numbers scroll up.",
                    },
                    "shift": {
                        "type": "boolean",
                        "description": "Optional for `action=scroll`. Hold shift while scrolling. Defaults to false.",
                    },
                    "time": {
                        "type": "number",
                        "description": "Required only by `action=wait`.",
                    },
                    "status": {
                        "type": "string",
                        "description": "Required only by `action=terminate`.",
                        "enum": ["success", "failure"],
                    },
                    "press_keys": {
                        "type": "array",
                        "description": "Required only by `action=hold_and_press`.",
                    },
                    "app_or_filename": {
                        "type": "string",
                        "description": "Required only by `action=open`.",
                    },
                    "app_code": {
                        "type": "string",
                        "description": "Required only by `action=switch_applications`.",
                    },
                    "cell_values": {
                        "type": "object",
                        "description": "Required only by `action=set_cell_values`.",
                    },
                    "app_name": {
                        "type": "string",
                        "description": "Required only by `action=set_cell_values`.",
                    },
                    "sheet_name": {
                        "type": "string",
                        "description": "Required only by `action=set_cell_values`.",
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Optional for `action=type`. Defaults to false.",
                    },
                    "enter": {
                        "type": "boolean",
                        "description": "Optional for `action=type`. Defaults to false.",
                    },
                },
            },
        },
    }


def build_base_system_prompt(tools_def: Dict, collapse_text: str) -> str:
    return (
        "You are a multi-purpose intelligent assistant. Based on my requests, you can use tools to help me complete various tasks.\n\n"
        "# Tools\n\n"
        "You have access to the following functions:\n\n"
        "<tools>\n"
        + json.dumps(tools_def)
        + "\n</tools>\n\n"
        "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
        "<tool_call>\n"
        "<function=example_function_name>\n"
        "<parameter=example_parameter_1>\n"
        "value_1\n"
        "</parameter>\n"
        "<parameter=example_parameter_2>\n"
        "This is the value for the second parameter\n"
        "that can span\n"
        "multiple lines\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n\n"
        "<IMPORTANT>\n"
        "Reminder:\n"
        "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
        "- Required parameters MUST be specified\n"
        "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
        "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
        f"- The current date is {datetime.today().strftime('%A, %B %d, %Y')}.\n"
        f"- Collapsed screenshots appear as text: {collapse_text}\n"
        "</IMPORTANT>\n\n"
        "# Response format\n\n"
        "Response format for every step:\n"
        "1) Action: a short imperative describing what to do in the UI.\n"
        "2) A single <tool_call>...</tool_call> block.\n\n"
        "Rules:\n"
        "- Output exactly in the order: Action, <tool_call>.\n"
        "- Be brief: one sentence for Action.\n"
        "- Do not output anything else outside those parts.\n"
        "- If finishing, use action=terminate in the tool call."
    )


def build_internal_system_prompt(tools_def: Dict, collapse_text: str) -> str:
    return (
        "You are a multi-purpose intelligent assistant. Based on my requests, you can use tools to help me complete various tasks.\n\n"
        "# Tools\n\n"
        "You have access to the following functions:\n\n"
        "<tools>\n"
        + json.dumps(tools_def, ensure_ascii=False)
        + "\n</tools>\n\n"
        "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
        "<tool_call>\n"
        "<function=example_function_name>\n"
        "<parameter=example_parameter_1>\n"
        "value_1\n"
        "</parameter>\n"
        "<parameter=example_parameter_2>\n"
        "This is the value for the second parameter\n"
        "that can span\n"
        "multiple lines\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n\n"
        "<IMPORTANT>\n"
        "Reminder:\n"
        "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
        "- Required parameters MUST be specified\n"
        "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
        "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
        f"- The current date is {datetime.today().strftime('%A, %B %d, %Y')}.\n"
        f"- Collapsed screenshots appear as text: {collapse_text}\n"
        "</IMPORTANT>\n\n"
        "# Response format\n\n"
        "For normal UI interaction steps:\n"
        "1) Action: a short imperative describing what to do in the UI.\n"
        "2) A single <tool_call>...</tool_call> block.\n\n"
        "For terminal steps, you may either:\n"
        "- output a final natural-language response with no tool call, or\n"
        "- use a terminal tool call such as call_user or terminate.\n\n"
        "Rules:\n"
        "- For non-terminal UI steps, output exactly in the order: Action, <tool_call>.\n"
        "- Be brief: one sentence for Action.\n"
        "- Do not output anything after a tool call.\n"
        "- Use call_user when you need user information or confirmation.\n"
        "- Use terminate when you want to explicitly end the task with a success or failure status.\n"
        "- If the task is infeasible, say so explicitly in the response."
    )


def build_hybrid_macro_system_prompt(tools_def: Dict, collapse_text: str) -> str:
    return (
        "You are a multi-purpose intelligent assistant. Based on my requests, you can use tools to help me complete various tasks.\n\n"
        "# Tools\n\n"
        "You have access to the following functions:\n\n"
        "<tools>\n"
        + json.dumps(tools_def, ensure_ascii=False)
        + "\n</tools>\n\n"
        "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
        "<tool_call>\n"
        "<function=computer_use>\n"
        "<parameter=action>\n"
        "one_of_the_allowed_action_names\n"
        "</parameter>\n"
        "<parameter=action_parameter_1>\n"
        "This is the value for the first parameter for specified action\n"
        "that can span\n"
        "multiple lines\n"
        "</parameter>\n"
        "<parameter=action_parameter_2>\n"
        "This is the value for the second parameter for specified action\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n\n"
        "<IMPORTANT>\n"
        "Reminder:\n"
        "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
        "- Required parameters MUST be specified\n"
        "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
        "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
        f"- The current date is {datetime.today().strftime('%A, %B %d, %Y')}.\n"
        f"- Collapsed screenshots appear as text: {collapse_text}\n"
        "</IMPORTANT>\n\n"
        "# Response format\n\n"
        "For normal UI interaction steps:\n"
        "1) Action: a short imperative describing what to do in the UI.\n"
        "2) A single <tool_call>...</tool_call> block.\n\n"
        "For terminal steps, you may either:\n"
        "- output a final natural-language response with no tool call, or\n"
        "- use a terminal tool call such as call_user or terminate.\n\n"
        "Rules:\n"
        "- For non-terminal UI steps, output exactly in the order: Action, <tool_call>.\n"
        "- Be brief: one sentence for Action.\n"
        "- Do not output anything after a tool call.\n"
        "- Prefer the exposed primitive and macro actions over legacy action names.\n"
        "- Use click for visible click targets instead of splitting into move and click.\n"
        "- Use type for text entry and scroll for wheel scrolling.\n"
        "- Use drag_and_drop for drag operations and set_cell_values for spreadsheet batch writes.\n"
        "- Use call_user when you need user information or confirmation.\n"
        "- Use terminate when you want to explicitly end the task with a success or failure status.\n"
        "- If the task is infeasible, say so explicitly in the response."
    )


def build_instruction_prompt(instruction: str, previous_actions_str: str) -> str:
    return (
        "\nPlease generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
        f"Instruction: {instruction}\n\n"
        f"Previous actions:\n"
        f"{previous_actions_str}"
    )
