from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analysis")
def analysis():
    return render_template("analysis.html")

@app.route("/price-prediction")
def price_prediction():
    return render_template("price_prediction.html")

if __name__ == "__main__":
    app.run(debug=True)
