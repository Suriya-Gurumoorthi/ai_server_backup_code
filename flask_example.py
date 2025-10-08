from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "ultravox-v0_5-llama-3_2-1b loaded"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
