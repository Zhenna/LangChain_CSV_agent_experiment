from agent import agent_executor, context

# CLI chat loop
print("\n🤖 CSV Chatbot (Offline CLI Mode)\nType 'exit' to quit.")
while True:
    user_input = input("\nUser: ")
    if user_input.lower() in {"exit", "quit"}:
        break
    try:
        response = agent_executor.invoke(context + "\n\n" + user_input)
        print("Bot:", response)
    except Exception as e:
        print("❌ Error:", str(e))