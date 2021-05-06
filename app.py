from flask import Flask, render_template, request
import seg as s
import seg_color as s_color

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
    mask_color = request.form['maskcolor']
    f.save('./static/' + f.filename)
    if mask_color=='rgb':
        seg_file = s_color.run_seg('./static/' + f.filename)
    else:
        seg_file = s.run_seg('./static/' + f.filename)
    return seg_file


if __name__ == "__main__":
    app.run(debug=True)
