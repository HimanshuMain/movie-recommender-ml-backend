from flask import Flask, request, jsonify
from recommender import recommend
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Movie Recommendation ML API is running"

@app.route("/recommend", methods=["GET"])
def get_recommendations():
    title = request.args.get("title")
    recommendations = recommend(title)
    return jsonify({"recommendations": recommendations})

if __name__ == "__main__":
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
