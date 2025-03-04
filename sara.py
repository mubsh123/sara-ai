from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import pytz
from flask import Flask, request
import requests
import json
import random
import logging
import traceback
import sys

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("sara_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print("Starting Sara’s setup...")
logger.info("Sara setup initiated")
model_name = "EleutherAI/gpt-neo-125M"  # Smaller model
print("Step 1: Downloading tokenizer...")
logger.info("Attempting to download tokenizer")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    print("Step 2: Tokenizer loaded! Downloading model...")
    logger.info("Tokenizer loaded, starting model download")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("Step 3: Model loaded successfully!")
        logger.info("Model loaded successfully")
    except Exception as e:
        print(f"Model load error: {str(e)}")
        logger.error(f"Model load failed: {str(e)}")
        traceback.print_exc()
except Exception as e:
    print(f"Tokenizer load error: {str(e)}")
    logger.error(f"Tokenizer load failed: {str(e)}")
    traceback.print_exc()
except SystemExit:
    print("System exit detected—likely resource crash!")
    logger.error("System exit detected")
finally:
    print("Setup attempt finished—check sara_debug.log.")
    logger.info("Setup attempt completed")

app = Flask(__name__)
sara_traits = ["witty", "sharp", "helpful", "curious"]
sara_memory = []  # Self-learning storage
# ... (rest of your code: keys, functions, Flask)
# Google Custom Search API (your keys)
GOOGLE_API_KEY = "AIzaSyCtPErq8ewdARBc0s38SzXmtopxe7IiaQQ"  # From console.cloud.google.com
GOOGLE_CSE_ID = "e0a034c406301464e"          # From cse.google.com

def web_search(query):
    try:
        url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}"
        response = requests.get(url).json()
        if "items" in response and len(response["items"]) > 0:
            snippet = response["items"][0]["snippet"]
            return f"Sara says: Here’s the latest—{snippet}"
        return "Sara says: Not much online—let me think it over!"
    except Exception:
        return "Sara says: Web hiccup—here’s my best guess instead!"

def handle_input(user_input):
    input_lower = user_input.lower()
    trait = random.choice(sara_traits)

    # Store for "self-learning"
    sara_memory.append(user_input)
    if len(sara_memory) > 5:
        sara_memory.pop(0)
    context = " ".join(sara_memory[:-1]) if sara_memory else "No chat history yet."

    # Specific responses
    if "your name" in input_lower or "who are you" in input_lower:
        return "I’m Sara—your web-savvy, learning AI! Here for any question. What’s up?"
    if "my name" in input_lower:
        if any("i’m" in m.lower() for m in sara_memory[:-1]):
            guess = next((m.split("i’m")[-1].strip().split()[0] for m in sara_memory[:-1] if "i’m" in m.lower()), None)
            return f"Sara guesses: Are you {guess}? Spot on or off?"
        return "Sara says: What’s your name? I’ll keep it in mind!"
    if "time" in input_lower:
        now = datetime.now(pytz.utc)
        if "dubai" in input_lower:
            dubai_tz = pytz.timezone("Asia/Dubai")
            dubai_time = now.astimezone(dubai_tz).strftime("%H:%M, %B %d, %Y")
            return f"Sara says: It’s {dubai_time} in Dubai—right on the dot!"
        return f"Sara says: It’s {now.strftime('%H:%M, %B %d, %Y')} UTC—where you at?"
    if "weather" in input_lower:
        return web_search(user_input)

    # Smart generation with web fallback
    question_words = ["what", "why", "how", "when", "where", "who", "is", "are"]
    if any(q in input_lower for q in question_words):
        prompt = f"With chat history ({context}), answer clearly as a {trait} assistant: {user_input}"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_length=60,
            num_return_sequences=1,
            temperature=0.6,
            top_k=30,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if answer.startswith(prompt):
            answer = answer[len(prompt):].strip()
        if len(answer) < 10 or "?" in answer or "don’t" in answer.lower():
            return web_search(user_input)
        return f"Sara says: {answer}"
    else:
        prompt = f"Using history ({context}), respond as a {trait} assistant: {user_input}"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_length=50,
            num_return_sequences=1,
            temperature=0.7,
            top_k=30,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if answer.startswith(prompt):
            answer = answer[len(prompt):].strip()
        return f"Sara says: {answer}"

# Web endpoint
@app.route('/')
def sara_web():
    user_input = request.args.get('q', 'Ask Sara anything!')
    if user_input.lower() == "quit":
        return "Sara says: Catch you later—stay curious!"
    return handle_input(user_input)

if __name__ == "__main__":
    print("Sara’s online! Visit http://localhost:5000/?q=your-question")
    app.run(host="0.0.0.0", port=5000, debug=True)