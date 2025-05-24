import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

csv_file = "data/wide_to_long_inverters.csv"
df = pd.read_csv(csv_file)

def describe_column(col):
    dtype = df[col].dtype
    sample = df[col].dropna().unique()[:3]
    samples = ", ".join(map(str, sample))
    return f"- `{col}` (type: {dtype}): e.g. {samples}"

column_descriptions = "\n".join(describe_column(col) for col in df.columns)

context = f"""
You are a data analyst working on `{csv_file}`.
The dataset contains the following columns:\n{column_descriptions}
Use this information to answer questions clearly. Avoid code in your answers.
"""

llm = OllamaLLM(model="llama3.2")

agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    allow_dangerous_code=True,
    max_iterations=15,  # 🔼 default is 5
    max_execution_time=60  # 🔼 seconds (optional)
)
