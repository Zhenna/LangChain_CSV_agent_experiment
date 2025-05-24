from fastapi import FastAPI
from schema import QueryRequest
from agent import agent, context

app = FastAPI()

@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        full_prompt = context + "\n\n" + request.question
        answer = agent.run(full_prompt)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}
