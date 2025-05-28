import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.agents import initialize_agent, AgentExecutor
from langchain.agents.agent_types import AgentType


# from pathlib import Path
# csv_file = Path("data") / "wide_to_long_inverters_processed.csv"

csv_file = "data/wide_to_long_inverters_processed.csv" 
df = pd.read_csv(csv_file)

def describe_column(col):
    dtype = df[col].dtype
    sample = df[col].dropna().unique()[:3]
    samples = ", ".join(map(str, sample))
    return f"- `{col}` (type: {dtype}): e.g. {samples}"

column_descriptions = "\n".join(describe_column(col) for col in df.columns)

tool_name = "python_repl_ast"

context = f"""
You are a data analyst working on `{csv_file}`.
The dataset contains the following columns:\n{column_descriptions}

Use the tool `{tool_name}` to perform all Python code execution. 
Do **not** use tools named 'filter', 'transform', 'query', or anything else — only `{tool_name}`.
Only respond with the correct tool name `{tool_name}` when choosing actions.

Avoid importing or redefining `df`. It is already available.

When filtering:
- Use expressions like `df[df['column'].str.contains("2024-01")]`
- Wrap conditions in parentheses when using `&` or `|`.
- Use `.fillna()` or proper comparisons if the column might contain missing values.
- Avoid writing code in your final answer. Just reason clearly.

Use this information to analyze or filter the dataset correctly.
"""


# Step 1: LLM
llm = OllamaLLM(model="llama3.2") #"mistral:instruct") #

# Step 2: Tools
    
class SafePythonTool(PythonAstREPLTool):
    def run(self, query: str, verbose: bool = False, **kwargs) -> str:
        # Optional: check for verbosity or color in kwargs if needed
        if "Path" in query or "read_csv" in query:
            return "❌ You should not redefine or reload the DataFrame. Just use `df`."
        return super().run(query,  verbose=verbose, **kwargs)
    
# class SafePythonTool(PythonAstREPLTool):
#     def run(self, query: str, verbose: bool = False) -> str:
#         if "Path" in query or "read_csv" in query:
#             return "❌ You should not redefine or reload the DataFrame. Just use `df`."
#         return super().run(query, verbose=verbose)

python_tool = SafePythonTool(locals={"df": df})

# python_tool = PythonAstREPLTool(locals={"df": df})

# Step 3: Agent
agent = initialize_agent(
    tools=[python_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    allow_dangerous_code=True,
    max_iterations=15,
    max_execution_time=60
)

# Step 4: Manually enable parsing error recovery
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent.agent,
    tools=agent.tools,
    llm=llm,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=15,
    max_execution_time=60,
    allow_dangerous_code=True
)
