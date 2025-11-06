# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import time
from typing import Literal
from typing import Optional

from google.adk.tools.computer_use.base_computer import BaseComputer
from google.adk.tools.computer_use.base_computer import ComputerEnvironment
from google.adk.tools.computer_use.base_computer import ComputerState
from playwright.async_api import async_playwright
import termcolor
from typing_extensions import override

# Define a mapping from the user-friendly key names to Playwright's expected key names.
# Playwright is generally good with case-insensitivity for these, but it's best to be canonical.
# See: https://playwright.dev/docs/api/class-keyboard#keyboard-press
# Keys like 'a', 'b', '1', '$' are passed directly.
PLAYWRIGHT_KEY_MAP = {
    "backspace": "Backspace",
    "tab": "Tab",
    "return": "Enter",  # Playwright uses 'Enter'
    "enter": "Enter",
    "shift": "Shift",
    "control": "Control",  # Or 'ControlOrMeta' for cross-platform Ctrl/Cmd
    "alt": "Alt",
    "escape": "Escape",
    "space": "Space",  # Can also just be " "
    "pageup": "PageUp",
    "pagedown": "PageDown",
    "end": "End",
    "home": "Home",
    "left": "ArrowLeft",
    "up": "ArrowUp",
    "right": "ArrowRight",
    "down": "ArrowDown",
    "insert": "Insert",
    "delete": "Delete",
    "semicolon": ";",  # For actual character ';'
    "equals": "=",  # For actual character '='
    "multiply": "Multiply",  # NumpadMultiply
    "add": "Add",  # NumpadAdd
    "separator": "Separator",  # Numpad specific
    "subtract": "Subtract",  # NumpadSubtract, or just '-' for character
    "decimal": "Decimal",  # NumpadDecimal, or just '.' for character
    "divide": "Divide",  # NumpadDivide, or just '/' for character
    "f1": "F1",
    "f2": "F2",
    "f3": "F3",
    "f4": "F4",
    "f5": "F5",
    "f6": "F6",
    "f7": "F7",
    "f8": "F8",
    "f9": "F9",
    "f10": "F10",
    "f11": "F11",
    "f12": "F12",
    "command": "Meta",  # 'Meta' is Command on macOS, Windows key on Windows
}


class PlaywrightComputer(BaseComputer):
  """Computer that controls Chromium via Playwright."""

  def __init__(
      self,
      screen_size: tuple[int, int],
      initial_url: str = "https://www.google.com",
      search_engine_url: str = "https://www.google.com",
      highlight_mouse: bool = False,
      user_data_dir: Optional[str] = None,
  ):
    self._initial_url = initial_url
    self._screen_size = screen_size
    self._search_engine_url = search_engine_url
    self._highlight_mouse = highlight_mouse
    self._user_data_dir = user_data_dir

  @override
  async def initialize(self):
    print("Creating session...")
    self._playwright = await async_playwright().start()

    # Define common arguments for both launch types
    browser_args = [
        "--disable-blink-features=AutomationControlled",
        "--disable-gpu",
    ]

    if self._user_data_dir:
      termcolor.cprint(
          f"Starting playwright with persistent profile: {self._user_data_dir}",
          color="yellow",
          attrs=["bold"],
      )
      # Use a persistent context if user_data_dir is provided
      self._context = await self._playwright.chromium.launch_persistent_context(
          self._user_data_dir,
          headless=False,
          args=browser_args,
      )
      self._browser = self._context.browser
    else:
      termcolor.cprint(
          "Starting playwright with a temporary profile.",
          color="yellow",
          attrs=["bold"],
      )
      # Launch a temporary browser instance if user_data_dir is not provided
      self._browser = await self._playwright.chromium.launch(
          args=browser_args,
          headless=False,
      )
      self._context = await self._browser.new_context()

    if not self._context.pages:
      self._page = await self._context.new_page()
      await self._page.goto(self._initial_url)
    else:
      self._page = self._context.pages[0]  # Use existing page if any

    await self._page.set_viewport_size({
        "width": self._screen_size[0],
        "height": self._screen_size[1],
    })
    termcolor.cprint(
        f"Started local playwright.",
        color="green",
        attrs=["bold"],
    )

  @override
  async def environment(self):
    return ComputerEnvironment.ENVIRONMENT_BROWSER

  @override
  async def close(self, exc_type, exc_val, exc_tb):
    if self._context:
      self._context.close()
    try:
      self._browser.close()
    except Exception as e:
      # Browser was already shut down because of SIGINT or such.
      if (
          "Browser.close: Connection closed while reading from the driver"
          in str(e)
      ):
        pass
      else:
        raise

    self._playwright.stop()

  async def open_web_browser(self) -> ComputerState:
    return await self.current_state()

  async def click_at(self, x: int, y: int):
    await self.highlight_mouse(x, y)
    await self._page.mouse.click(x, y)
    await self._page.wait_for_load_state()
    return await self.current_state()

  async def hover_at(self, x: int, y: int):
    await self.highlight_mouse(x, y)
    await self._page.mouse.move(x, y)
    await self._page.wait_for_load_state()
    return await self.current_state()

  async def type_text_at(
      self,
      x: int,
      y: int,
      text: str,
      press_enter: bool = True,
      clear_before_typing: bool = True,
  ) -> ComputerState:
    await self.highlight_mouse(x, y)
    await self._page.mouse.click(x, y)
    await self._page.wait_for_load_state()

    if clear_before_typing:
      await self.key_combination(["Control", "A"])
      await self.key_combination(["Delete"])

    await self._page.keyboard.type(text)
    await self._page.wait_for_load_state()

    if press_enter:
      await self.key_combination(["Enter"])
    await self._page.wait_for_load_state()
    return await self.current_state()

  async def _horizontal_document_scroll(
      self, direction: Literal["left", "right"]
  ) -> ComputerState:
    # Scroll by 50% of the viewport size.
    horizontal_scroll_amount = await self.screen_size()[0] // 2
    if direction == "left":
      sign = "-"
    else:
      sign = ""
    scroll_argument = f"{sign}{horizontal_scroll_amount}"
    # Scroll using JS.
    await self._page.evaluate(f"window.scrollBy({scroll_argument}, 0); ")
    await self._page.wait_for_load_state()
    return await self.current_state()

  async def scroll_document(
      self, direction: Literal["up", "down", "left", "right"]
  ) -> ComputerState:
    if direction == "down":
      return await self.key_combination(["PageDown"])
    elif direction == "up":
      return await self.key_combination(["PageUp"])
    elif direction in ("left", "right"):
      return await self._horizontal_document_scroll(direction)
    else:
      raise ValueError("Unsupported direction: ", direction)

  async def scroll_at(
      self,
      x: int,
      y: int,
      direction: Literal["up", "down", "left", "right"],
      magnitude: int,
  ) -> ComputerState:
    await self.highlight_mouse(x, y)

    await self._page.mouse.move(x, y)
    await self._page.wait_for_load_state()

    dx = 0
    dy = 0
    if direction == "up":
      dy = -magnitude
    elif direction == "down":
      dy = magnitude
    elif direction == "left":
      dx = -magnitude
    elif direction == "right":
      dx = magnitude
    else:
      raise ValueError("Unsupported direction: ", direction)

    await self._page.mouse.wheel(dx, dy)
    await self._page.wait_for_load_state()
    return await self.current_state()

  async def wait(self, seconds: int) -> ComputerState:
    await asyncio.sleep(seconds)
    return await self.current_state()

  async def go_back(self) -> ComputerState:
    await self._page.go_back()
    await self._page.wait_for_load_state()
    return await self.current_state()

  async def go_forward(self) -> ComputerState:
    await self._page.go_forward()
    await self._page.wait_for_load_state()
    return await self.current_state()

  async def search(self) -> ComputerState:
    return await self.navigate(self._search_engine_url)

  async def navigate(self, url: str) -> ComputerState:
    await self._page.goto(url)
    await self._page.wait_for_load_state()
    return await self.current_state()

  async def key_combination(self, keys: list[str]) -> ComputerState:
    # Normalize all keys to the Playwright compatible version.
    keys = [PLAYWRIGHT_KEY_MAP.get(k.lower(), k) for k in keys]

    for key in keys[:-1]:
      await self._page.keyboard.down(key)

    await self._page.keyboard.press(keys[-1])

    for key in reversed(keys[:-1]):
      await self._page.keyboard.up(key)

    return await self.current_state()

  async def drag_and_drop(
      self, x: int, y: int, destination_x: int, destination_y: int
  ) -> ComputerState:
    await self.highlight_mouse(x, y)
    await self._page.mouse.move(x, y)
    await self._page.wait_for_load_state()
    await self._page.mouse.down()
    await self._page.wait_for_load_state()

    await self.highlight_mouse(destination_x, destination_y)
    await self._page.mouse.move(destination_x, destination_y)
    await self._page.wait_for_load_state()
    await self._page.mouse.up()
    return await self.current_state()

  async def current_state(self) -> ComputerState:
    await self._page.wait_for_load_state()
    # Even if Playwright reports the page as loaded, it may not be so.
    # Add a manual sleep to make sure the page has finished rendering.
    time.sleep(0.5)
    screenshot_bytes = await self._page.screenshot(type="png", full_page=False)
    return ComputerState(screenshot=screenshot_bytes, url=self._page.url)

  async def screen_size(self) -> tuple[int, int]:
    return self._screen_size

  async def highlight_mouse(self, x: int, y: int):
    if not self._highlight_mouse:
      return
    await self._page.evaluate(f"""
        () => {{
            const element_id = "playwright-feedback-circle";
            const div = document.createElement('div');
            div.id = element_id;
            div.style.pointerEvents = 'none';
            div.style.border = '4px solid red';
            div.style.borderRadius = '50%';
            div.style.width = '20px';
            div.style.height = '20px';
            div.style.position = 'fixed';
            div.style.zIndex = '9999';
            document.body.appendChild(div);

            div.hidden = false;
            div.style.left = {x} - 10 + 'px';
            div.style.top = {y} - 10 + 'px';

            setTimeout(() => {{
                div.hidden = true;
            }}, 2000);
        }}
    """)
    # Wait a bit for the user to see the cursor.
    time.sleep(1)
