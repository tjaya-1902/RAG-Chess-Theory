from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from retriever import retrieve_chunks
from generator import generate_answer

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)  

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get("question", "")
    
    if not question:
        return jsonify({"error": "Question is required"}), 400

    chunks = retrieve_chunks(question)
    context = " ".join(chunks["content"].tolist())
    
    answer = generate_answer(question, context)
    
    return jsonify({"answer": answer, "context": context})

@app.route('/analyze_opening', methods=['POST'])
def analyze_opening():
    data = request.json
    moves = data.get("moves", [])

    if not moves:
        return jsonify({"error": "Move list is empty"}), 400

    from retriever import find_opening_by_moves
    topic, description = find_opening_by_moves(moves)
    
    return jsonify({
        "topic": topic,
        "description": description
    })

if __name__ == '__main__':
    app.run(debug=True)