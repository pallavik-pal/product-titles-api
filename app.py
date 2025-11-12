from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
<<<<<<< HEAD
import os

app = Flask(__name__)

MODEL_REPO = "pallaviik/product-titles-gpt_2"
model = None
tokenizer = None
device = "cpu"  # Force CPU to save memory


def load_model():
    """Lazy-load model only when needed"""
    global model, tokenizer
    if model is None or tokenizer is None:
        print("Loading model from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
        model = AutoModelForCausalLM.from_pretrained(MODEL_REPO)
        model.to(device)
        print("✅ Model loaded successfully!")


@app.route("/")
def home():
    return "✅ Flask NLP API is running!"


@app.route("/generate", methods=["POST"])
def generate_api():
    load_model()
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=40,
        num_return_sequences=3,
=======

app = Flask(__name__)

MODEL_REPO = "pallaviik/product-titles-gpt_2"   # your HF repo

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = AutoModelForCausalLM.from_pretrained(MODEL_REPO)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_next_words(prompt, num_sequences=3, max_length=15):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=num_sequences,
>>>>>>> 7d06cd704ef8f8cc70f25594334da610fced5331
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )
<<<<<<< HEAD
    preds = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    return jsonify({"prompt": prompt, "predictions": preds})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
=======
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

@app.route("/")
def home():
    return "✅ NLP API running successfully!"

@app.route("/generate", methods=["POST"])
def generate_api():
    data = request.get_json()
    prompt = data.get("prompt", "")
    preds = generate_next_words(prompt)
    return jsonify({"prompt": prompt, "predictions": preds})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
>>>>>>> 7d06cd704ef8f8cc70f25594334da610fced5331
