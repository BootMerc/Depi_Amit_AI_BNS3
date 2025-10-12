import json
import os
from response import get_response

# Load chatbot.json (relative path)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "chatbot.json"), "r", encoding="utf-8") as file:
    responses = json.load(file)

if __name__ == "__main__":
    print("Chatbot: Hi! How can I assist you today? (type 'quit' to exit)")

    while True:
        user_input = input("You: ").strip().lower()

        # Exit condition
        if user_input in ["quit", "exit", "bye", "goodbye"]:
            print("Chatbot: Goodbye! Have a nice day.")
            break

        # Get chatbot response
        response = get_response(user_input, responses)
        print("Chatbot:", response)
