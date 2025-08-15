from flask import Flask, render_template, request, jsonify
from services.search_service import semantic_search_json

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/api/search', methods=['GET'])
def api_search():
    query_text = request.args.get('q', '')  # รับค่าจาก URL: /api/search?q=xxx
    results = semantic_search_json(query_text, threshold=0.6)
    return jsonify(results)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3025)
