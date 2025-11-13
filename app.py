
from flask import Flask, request, jsonify
import requests
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

HF_MODEL = "pallaviik/product-titles-gpt_2"
HF_TOKEN = "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"

@app.route('/')
def home():
    return jsonify({"message": "Product Title API is running!", "status": "success"})

@app.route('/api/generate', methods=['POST'])
def generate_titles():
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({"error": "Prompt is required"}), 400
            
        prompt = data['prompt']
        
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 40,
                "num_return_sequences": 3,
                "do_sample": True,
                "temperature": 0.8,
                "top_p": 0.9
            }
        }
        
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers=headers,
            json=payload,
            timeout=45
        )
        
        if response.status_code == 200:
            results = response.json()
            
            titles = []
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict) and 'generated_text' in item:
                        titles.append(item['generated_text'])
                    else:
                        titles.append(str(item))
            else:
                titles = [str(results)]
                
            return jsonify({
                "success": True,
                "prompt": prompt,
                "titles": titles
            })
            
        elif response.status_code == 503:
            return jsonify({
                "error": "Model is loading, please try again in 30 seconds",
                "status": "loading"
            }), 503
            
        else:
            return jsonify({
                "error": f"API error: {response.status_code}",
                "details": response.text
            }), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "service": "product-title-api"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
