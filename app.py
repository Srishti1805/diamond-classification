from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved scaler and model
scaler = joblib.load("scaler.pkl")
svm_model = joblib.load("svm_model.pkl")

shape_mapping = {0: 'Cushion', 1: 'Cushion Modified', 2: 'Emerald', 3: 'Heart', 4: 'Marquise', 5: 'Oval', 6: 'Pear', 7: 'Princess', 8: 'Radiant', 9: 'Round', 10: 'Square Radiant'}
color_mapping = {0: 'D', 1: 'E', 2: 'F', 3: 'G', 4: 'H'}
clarity_mapping = {0: 'FL', 1: 'IF', 2: 'VS1', 3: 'VS2', 4: 'VVS1', 5: 'VVS2'}
polish_mapping = {0: 'Excellent', 1: 'Good', 2: 'Very Good'}
symmetry_mapping = {0: 'Excellent', 1: 'Good', 2: 'Very Good'}
girdle_mapping = {0: 'Extremely Thick', 1: 'Extremely Thin to Extremely Thick', 2: 'Extremely Thin to Medium', 
                  3: 'Extremely Thin to Slightly Thick', 4: 'Medium', 5: 'Medium to Extremely Thick', 6: 'Medium to Slightly Thick', 
                  7: 'Medium to Thick', 8: 'Medium to Very Thick', 9: 'Slightly Thick', 10: 'Slightly Thick to Extremely Thick', 
                  11: 'Slightly Thick to Slightly Thick', 12: 'Slightly Thick to Thick', 13: 'Slightly Thick to Very Thick', 
                  14: 'Thick', 15: 'Thick to Extremely Thick', 16: 'Thick to Very Thick', 17: 'Thin', 18: 'Thin to Extremely Thick', 
                  19: 'Thin to Medium', 20: 'Thin to Slightly Thick', 21: 'Thin to Thick', 22: 'Thin to Very Thick', 23: 'Very Thick', 
                  24: 'Very Thick to Extremely Thick', 25: 'Very Thin to Extremely Thick', 26: 'Very Thin to Slightly Thick', 
                  27: 'Very Thin to Thick', 28: 'Very Thin to Very Thick'}


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            # Retrieve categorical and numerical inputs
            shape = int(request.form["shape"])
            color = int(request.form["color"])
            clarity = int(request.form["clarity"])
            carat_weight = float(request.form["carat_weight"])
            length_width_ratio = float(request.form["length_width_ratio"])
            depth = float(request.form["depth"])
            table = float(request.form["table"])
            polish = int(request.form["polish"])
            symmetry = int(request.form["symmetry"])
            girdle = int(request.form["girdle"])
            length = float(request.form["length"])
            width = float(request.form["width"])
            height = float(request.form["height"])
            price = float(request.form["price"])

            # Prepare input for prediction
            features = np.array([[shape, color, clarity, carat_weight, length_width_ratio, depth, table, 
                                  polish, symmetry, girdle, length, width, height, price]])
            
            # Scale features
            scaled_features = scaler.transform(features)

            # Predict
            prediction = svm_model.predict(scaled_features)[0]
            prediction_map = {0: 'GIA', 1: 'GIA Lab-Grown', 2: 'IGI Lab-Grown'}
            prediction = prediction_map[prediction]

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction, shape_mapping=shape_mapping, color_mapping=color_mapping,
                           clarity_mapping=clarity_mapping, polish_mapping=polish_mapping, symmetry_mapping=symmetry_mapping,
                           girdle_mapping=girdle_mapping)


if __name__ == "__main__":
    app.run(debug=True, port=8080)
