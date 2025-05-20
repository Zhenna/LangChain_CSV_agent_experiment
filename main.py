# from langchain.agents import create_csv_agent
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain_community.llms import Ollama

# Step 1: Initialize local LLM (Ollama must be running locally)
llm = Ollama(model="llama3.2")  # mistral Replace with "llama2", "gemma", etc.

# Step 2: CSV file path
csv_file_path = "Overall inverter downtime 2024 .csv"

# Step 3: Create LangChain CSV Agent
agent = create_csv_agent(
    llm=llm,
    path=csv_file_path,
    verbose=True,
    agent_type="zero-shot-react-description",
    allow_dangerous_code=True
)

# Step 4: Run a natural language query
query = "How many unique values by Description? What are they?"
response = agent.run(query)

print(f"Query: {query} \n Answer: {response}")
