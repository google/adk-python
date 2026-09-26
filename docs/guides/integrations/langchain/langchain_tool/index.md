# LangchainTool

LangchainTool is an adapter that wraps Langchain tools for use within the ADK
framework. It converts Langchain tool schemas into a format compatible with
Google generative AI function calling.

## Introduction

The Langchain ecosystem provides a wide variety of pre-built tools for tasks
ranging from web searching to database interaction. The LangchainTool class
allows developers to integrate these existing tools into ADK agents without
rewriting the underlying logic or schema definitions.

This adapter manages the translation between Langchain conventions and the ADK
tool interface. It handles both synchronous and asynchronous tools, extracts
parameter schemas from Langchain StructuredTool instances, and respects
Langchain-specific behaviors like direct result returning.

## Get started

The following example demonstrates how to wrap a Langchain YouTube search tool
and provide it to an ADK agent.

```python
from google.adk.agents.llm_agent import Agent
from google.adk.integrations.langchain import LangchainTool
from langchain_community.tools.youtube.search import YouTubeSearchTool

# Instantiate the standard Langchain tool
langchain_yt_tool = YouTubeSearchTool()

# Wrap the tool for use in ADK
adk_yt_tool = LangchainTool(
    tool=langchain_yt_tool,
)

# Pass the wrapped tool to an agent
youtube_search_agent = Agent(
    name="youtube_search_agent",
    instruction="Search for singer names and video counts provided by the user.",
    tools=[adk_yt_tool],
)
```

## How it works

LangchainTool inherits from FunctionTool and acts as a bridge between the two
frameworks. When initialized, the adapter inspects the provided Langchain tool
to identify its execution method, which is typically named run or _run. The
adapter also extracts the tool name and description to build the function
declaration that the generative model sees.

During execution, the adapter maps the arguments provided by the model to the
expected inputs of the Langchain tool. If the wrapped tool has the
return_direct attribute set to True, the adapter automatically updates the tool
context to skip the summarization phase. This behavior ensures that the raw
output of the tool is returned to the user or the calling agent immediately,
matching Langchain's intended execution flow.

## Configuration options

The following options are available when configuring a LangchainTool through the
LangchainToolConfig class or a configuration file.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `tool` | `str` | | The fully qualified path of the Langchain tool instance. |
| `name` | `str` | `''` | The name of the tool. |
| `description` | `str` | `''` | The description of the tool. |

The tool option requires a string representing the fully qualified name of the
tool object so the ADK can resolve and instantiate it. The name and description
options allow developers to override the metadata defined within the Langchain
tool itself. Overriding these values is useful when the original tool
description does not provide enough context for the generative model to use the
tool effectively.

## Advanced applications

Developers can wrap Langchain StructuredTool instances to provide complex
parameter schemas to the model. LangchainTool automatically detects the
args_schema of a StructuredTool and uses it to build a detailed function
declaration.

```python
from google.adk.integrations.langchain import LangchainTool
from langchain_core.tools.structured import StructuredTool
from pydantic import BaseModel

class AddSchema(BaseModel):
    x: int
    y: int

def sync_add(x: int, y: int) -> int:
    return x + y

# Create a Langchain StructuredTool with a Pydantic schema
langchain_add_tool = StructuredTool.from_function(
    func=sync_add,
    name="add_numbers",
    description="Adds two integers together",
    args_schema=AddSchema,
)

# The adapter will preserve the x and y parameter definitions for the model
adk_add_tool = LangchainTool(tool=langchain_add_tool)
```

The adapter also handles error states specifically for direct-return tools. If a
tool is configured with return_direct=True but encounters an execution error,
the adapter will not skip summarization. This allows the generative model to
observe the error message and potentially attempt a corrected tool call.

## Limitations

The wrapped object must be a valid Langchain tool or an object that implements
the run or _run method. If the adapter cannot find a callable execution method
on the provided tool, it raises a ValueError during initialization.

## Related samples

- [a2a_auth](../../../../../contributing/samples/a2a/a2a_auth/agent.py) - Demonstrates using Langchain tools within a remote agent architecture.
- [langchain_structured_tool_agent](../../../../../contributing/samples/integrations/langchain_structured_tool_agent/agent.py) - Shows how to use Langchain StructuredTool with ADK agents.
- [langchain_youtube_search_agent](../../../../../contributing/samples/integrations/langchain_youtube_search_agent/agent.py) - A practical example of wrapping the Langchain YouTube search utility.
```

In []:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

```

In []:
```python
# Load the breast cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Display the first few rows of the dataset
print(X.head())

```

Out []:
```output
<output truncated>
```

In []:
```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

```

In []:
```python
# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

```

Out []:
```output
<pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br/>On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b><input/><label>LogisticRegression</label><pre>LogisticRegression()</pre>
```

In []:
```python
# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

```

Out []:
```output
Accuracy: 0.9737
Confusion Matrix:
[[41  2]
 [ 1 70]]
Classification Report:
              precision    recall  f1-score   support
           0       0.98      0.95      0.96        43
           1       0.97      0.99      0.98        71
    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114
```

In []:
```python
# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

```

Out []:
```output

```
