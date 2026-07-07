"""Flask web application serving the GenAI assistant.

POST /generate accepts {"message": ..., "model": "llama"|"granite"|"mistral"}
and returns structured JSON: summary, sentiment, response, next_step, duration.
"""

import time

from flask import Flask, request, jsonify, render_template

from model import llama_response, granite_response, mistral_response

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    user_message = data.get("message")
    model = data.get("model")

    if not user_message or not model:
        return jsonify({"error": "Missing message or model selection"}), 400

    # System prompt covers every field of the AIResponse schema,
    # including the exercise's next_step recommendation.
    system_prompt = (
        "You are an AI assistant helping with customer inquiries. "
        "Analyze the user's message and provide: a concise summary of their message, "
        "a sentiment score from 0 (very negative) to 100 (very positive), "
        "a helpful and concise suggested response to the user, "
        "and a recommended next step the support representative should take to resolve the issue."
    )

    start_time = time.time()

    try:
        if model == "llama":
            result = llama_response(system_prompt, user_message)
        elif model == "granite":
            result = granite_response(system_prompt, user_message)
        elif model == "mistral":
            result = mistral_response(system_prompt, user_message)
        else:
            return jsonify({"error": "Invalid model selection"}), 400

        result["duration"] = time.time() - start_time
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
