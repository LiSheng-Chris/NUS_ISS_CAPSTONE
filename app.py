from flask import Flask, render_template
import seg as s

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")
    
@app.route("/world")
def world():
    return "Hello, world"
    
@app.route("/seg")
def seg():
    seg_file=s.run_seg('./img/ZT80_38_A_1_13.jpg')
    return seg_file
    
if __name__ == "__main__":
    app.run(debug=True)