from flask import Flask, request, jsonify
from prj125 import getPrediction

app = Flask(__name__)
@app.route("/preditic-alpha", methods = ["POST"])

def predictData():
    image = request.files.get("alpha")
    prediction = getPrediction(image)
    
    return jsonify({
        "Prediction": prediction
    }), 200

if(__name__ == "__main__"):
    app.run(debug = True)