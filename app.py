from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )
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
