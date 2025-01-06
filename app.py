from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained machine learning model (model.pkl)
with open("Car-Price-Prediction-main/Machine Learning Final Proj/model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input from the form
    try:
        engine_hp = float(request.form['engine_hp'])  # Ensure the field names match
        year = int(request.form['year'])
        highway_mpg = float(request.form['highway_mpg'])
    except KeyError as e:
        return f"Missing field: {e}", 400  # Handle missing fields

    # Prepare the input data for prediction (3 features)
    input_data = np.array([[engine_hp, year, highway_mpg]])

    # Predict the price using the trained model
    predicted_price = model.predict(input_data)[0]

    # Return the result to the user
    return render_template('index.html', predicted_price=round(predicted_price, 2))

if __name__ == '__main__':
    app.run(debug=True)