import random

def get_response(user_input, responses):
    user_input = user_input.lower()

    # Loop through each category in JSON
    for category, data in responses.items():
        # Skip default since it has no patterns
        if category == "default":
            continue
        
        # Check patterns against user input
        for pattern in data["patterns"]:
            if pattern in user_input:
                return random.choice(data["responses"])

    # Fallback response
    return random.choice(responses["default"]["responses"])
import random
from datetime import datetime

def get_response(user_input, responses):
    user_input = user_input.lower()

    # Special case: time
    if any(word in user_input for word in ["time", "clock"]):
        current_time = datetime.now().strftime("%H:%M:%S")
        return f"The current time is {current_time}."

    # Special case: date
    if any(word in user_input for word in ["date", "day", "today"]):
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        return f"Today is {current_date}."

    # General pattern matching
    for category, data in responses.items():
        if category == "default":
            continue

        for pattern in data["patterns"]:
            if pattern in user_input:
                return random.choice(data["responses"])
    return random.choice(responses["default"]["responses"])
