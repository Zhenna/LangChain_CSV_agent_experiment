import pandas as pd
from langchain_ollama import OllamaLLM
# from langchain_experimental.tools import PandasDataFrameTool
# from langchain.tools.python import PythonREPLTool
# from langchain.tools.python import PythonAstREPLTool
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.agents import initialize_agent, AgentExecutor
from langchain.agents.agent_types import AgentType

df1 = pd.read_csv("data/wide_to_long_inverters.csv")
df2 = pd.read_csv("data/Raw data spansafe for CB.csv")
# df3 = pd.read_csv("data/Raw data spansafe for inverter.csv")

# def describe_column(col):
#     dtype = df[col].dtype
#     sample = df[col].dropna().unique()[:3]
#     samples = ", ".join(map(str, sample))
#     return f"- `{col}` (type: {dtype}): e.g. {samples}"

# column_descriptions = "\n".join(describe_column(col) for col in df.columns)

# context = f"""
# You are a data analyst working on `{csv_file}`.
# The dataset contains the following columns:\n{column_descriptions}
# To compare values across inverters within a specific time range, 
# you can aggregate by Inverter_ID and the appropriate time period,
# after converting timestamp column to DatetimeIndex.
# Use this information to answer questions clearly. Avoid code in your answers.
# """

# Create a shared context describing each DataFrame
def describe_df(df, name):
    return "\n".join(
        f"- `{name}.{col}` (type: {df[col].dtype}): e.g. {', '.join(map(str, df[col].dropna().unique()[:3]))}"
        for col in df.columns
    )


context = f"""
You are a data analyst. You have access to three datasets:

**df1: The inverter timeseries data **
{describe_df(df1, 'df1')}

To compare values across inverters within a specific time range, 
you can aggregate by Inverter_ID and the appropriate time period,
after converting timestamp column to DatetimeIndex.
Use this information to answer questions clearly. Avoid code in your answers.

**df2: The combiner box data**
{describe_df(df2, 'df2')}

Do not use this dataset if question only asks about inverters.

Refer to the datasets as df1, df2 in your analysis.
Avoid code in your answers.
"""



# Step 1: LLM
llm = OllamaLLM(model="mistral:instruct") #"llama3.2")

# Step 2: Tools
# python_tool = PythonAstREPLTool(locals={"df": df})
python_tool = PythonAstREPLTool(locals={
    "df1": df1,
    "df2": df2,
})

# Step 3: Agent
agent = initialize_agent(
    tools=[python_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
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
