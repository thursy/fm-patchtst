import os 
from flask import Flask, render_template, Response, request, jsonify

# from utils.rag import *
from .utils.rag import *

app = Flask(__name__, static_url_path='/static', template_folder='templates/')
    

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/rag")
def kontrak():
    return render_template('rag.html')


@app.route('/get_data', methods=['POST'])
def text2sql():
    data = request.get_json()
    user_question = data.get('user_question')
    answer = question_detail(user_question)
    print(answer)
    return jsonify(answer)


@app.route('/qna_wx', methods=['POST'])
def ask_wxai():
    data = request.get_json()
    answer, eta_wxai = query_wxai(data)
    response_data = {"answer": answer, "eta_wxai":eta_wxai}
    print(data)
    print(response_data)
    return jsonify(response_data)


@app.route('/qna_wx_stream', methods=['POST'])
def ask_wxai_stream():
    data = request.get_json()
    print(data)
    return Response(query_wxai(data, streaming=True), content_type='text/event-stream')


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))