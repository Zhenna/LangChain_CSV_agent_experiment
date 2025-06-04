import pandas as pd
import numpy as np
from langchain_ollama import OllamaLLM
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType

# === Configuration ===
CSV_PATH = "data/inverter_prediction.csv"
TOOL_NAME = "python_repl_ast"
MODEL_NAME = "llama3.2"

# === Load and Prepare Data ===
df = pd.read_csv(CSV_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# === Column Description for Prompt Context ===
def describe_column(col):
    dtype = df[col].dtype
    sample = df[col].dropna().unique()[:3]
    samples = ", ".join(map(str, sample))
    return f"- `{col}` (type: {dtype}): e.g. {samples}"

column_descriptions = "\n".join(describe_column(col) for col in df.columns)

# === Agent Context Prompt ===
context = f"""
🔁 WHEN AGGREGATING BY DAY OR HOUR:
- Convert `df['timestamp']` to datetime with `pd.to_datetime(df['timestamp'])`
- Use `df['timestamp'].dt.floor('h')` to group by hour
- Use `df['timestamp'].dt.date` to group by day
- Then use `groupby(...).mean()` to compute the average

‼️ Always follow these steps exactly when doing time-based aggregation.

You are a data analyst working on `{CSV_PATH}`.
The dataset contains the following columns:\n{column_descriptions}

Use the tool `{TOOL_NAME}` to perform all Python code execution. 
Do **not** use tools named 'filter', 'transform', 'query', or anything else — only `{TOOL_NAME}`.
Only respond with the correct tool name `{TOOL_NAME}` when choosing actions.

Avoid importing or redefining `df`. It is already available.

When filtering:
- Use expressions like `df[df['timestamp'].dt.strftime('%Y-%m').str.contains("2024-01")]`
- Wrap conditions in parentheses when using `&` or `|`.
- Use `.fillna()` or proper comparisons if the column might contain missing values.
- Avoid writing code in your final answer. Just reason clearly.
"""

# === LLM Setup ===
llm = OllamaLLM(model=MODEL_NAME)

# === Safe Python Execution Tool ===
class SafePythonTool(PythonAstREPLTool):
    def run(self, query: str, verbose: bool = False, **kwargs) -> str:
        if "Path" in query or "read_csv" in query:
            return "❌ You should not redefine or reload the DataFrame. Just use `df`."
        return super().run(query, verbose=verbose, **kwargs)

python_tool = SafePythonTool(locals={"df": df, "pd": pd, "np": np})

# === Agent Executor ===
agent_executor = initialize_agent(
    tools=[python_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    allow_dangerous_code=True,
    max_iterations=15,
    max_execution_time=60,
)
