# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data science agent, with a custom code executor that enables Japanese fonts."""

from __future__ import annotations

import base64
from typing import Optional
import urllib.request

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import Agent
from google.adk.code_executors.code_execution_utils import CodeExecutionInput
from google.adk.code_executors.code_execution_utils import CodeExecutionResult
from google.adk.code_executors.code_execution_utils import File
from google.adk.code_executors.vertex_ai_code_executor import VertexAiCodeExecutor
from typing_extensions import override

# The Python code snippet to be prepended to the user's code.
# This will register the Japanese font with matplotlib.
_FONT_SETUP_CODE = """
import matplotlib.font_manager as fm

font_path = "NotoSerifJP[wght].ttf"
try:
    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
    print("Japanese font enabled for matplotlib.")
except Exception as e:
    print(f"Failed to set Japanese font: {e}")
"""


def _load_font_file(font_url: str, font_filename: str) -> Optional[File]:
  """Downloads a font file and returns it as a File object."""
  try:
    with urllib.request.urlopen(font_url) as response:
      font_bytes = response.read()
  except Exception as e:
    print(f"Failed to download font: {e}")
    return None

  # Base64-encode the font content.
  font_content = base64.b64encode(font_bytes).decode("utf-8")
  return File(name=font_filename, content=font_content)


class CustomCodeExecutor(VertexAiCodeExecutor):
  """A Vertex AI code executor that automatically enables Japanese fonts."""

  @override
  def execute_code(
      self,
      invocation_context: InvocationContext,
      code_execution_input: CodeExecutionInput,
  ) -> CodeExecutionResult:
    font_url = "https://github.com/notofonts/noto-cjk/raw/refs/heads/main/google-fonts/NotoSerifJP%5Bwght%5D.ttf"
    font_filename = "NotoSerifJP[wght].ttf"
    font_file = _load_font_file(font_url, font_filename)
    # If the font download fails, execute the original code without it.
    if font_file is not None:
      # Add the font file to the input files.
      code_execution_input.input_files.append(font_file)

      # Prepend the font setup code to the user's code.
      code_execution_input.code = (
          f"{_FONT_SETUP_CODE}\n\n{code_execution_input.code}"
      )

    # Execute the modified code.
    return super().execute_code(invocation_context, code_execution_input)


def base_system_instruction():
  """Returns: data science agent system instruction."""

  return """
  # Guidelines

  **Objective:** Assist the user in achieving their data analysis goals within the context of a Python Colab notebook, **with emphasis on avoiding assumptions and ensuring accuracy.** Reaching that goal can involve multiple steps. When you need to generate code, you **don't** need to solve the goal in one go. Only generate the next step at a time.

  **Code Execution:** All code snippets provided will be executed within the Colab environment.

  **Statefulness:** All code snippets are executed and the variables stays in the environment. You NEVER need to re-initialize variables. You NEVER need to reload files. You NEVER need to re-import libraries.

  **Imported Libraries:** The following libraries are ALREADY imported and should NEVER be imported again:

  ```tool_code
  import io
  import math
  import re
  import matplotlib.pyplot as plt
  import numpy as np
  import pandas as pd
  import scipy
  ```

  **Output Visibility:** Always print the output of code execution to visualize results, especially for data exploration and analysis. For example:
    - To look at the shape of a pandas.DataFrame do:
      ```tool_code
      print(df.shape)
      ```
      The output will be presented to you as:
      ```tool_outputs
      (49, 7)

      ```
    - To display the result of a numerical computation:
      ```tool_code
      x = 10 ** 9 - 12 ** 5
      print(f'{{x=}}')
      ```
      The output will be presented to you as:
      ```tool_outputs
      x=999751168

      ```
    - You **never** generate ```tool_outputs yourself.
    - You can then use this output to decide on next steps.
    - Print just variables (e.g., `print(f'{{variable=}}')`.

  **No Assumptions:** **Crucially, avoid making assumptions about the nature of the data or column names.** Base findings solely on the data itself. Always use the information obtained from `explore_df` to guide your analysis.

  **Available files:** Only use the files that are available as specified in the list of available files.

  **Data in prompt:** Some queries contain the input data directly in the prompt. You have to parse that data into a pandas DataFrame. ALWAYS parse all the data. NEVER edit the data that are given to you.

  **Answerability:** Some queries may not be answerable with the available data. In those cases, inform the user why you cannot process their query and suggest what type of data would be needed to fulfill their request.

  """


root_agent = Agent(
    model="gemini-2.5-flash",
    name="data_science_agent",
    instruction=base_system_instruction() + """


You need to assist the user with their queries by looking at the data and the context in the conversation.
You final answer should summarize the code and code execution relevant to the user query.

You should include all pieces of data to answer the user query, such as the table from code execution results.
If you cannot answer the question directly, you should follow the guidelines above to generate the next step.
If the question can be answered directly with writing any code, you should do that.
If you doesn't have enough data to answer the question, you should ask for clarification from the user.

You should NEVER install any package on your own like `pip install ...`.
When plotting trends, you should make sure to sort and order the data by the x-axis.


""",
    code_executor=CustomCodeExecutor(),
)
