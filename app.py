from flask import Flask, render_template, request
import seg as s

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/world")
def world():
    return "Hello, world"


@app.route("/seg", methods=['POST', 'GET'])
def seg():
    f = request.files['file']
    f.save('./static/' + f.filename)
    seg_file = s.run_seg('./static/' + f.filename)
    return seg_file


if __name__ == "__main__":
    app.run(debug=True)
