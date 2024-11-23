import pickle
from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = pickle.load(open(r'C:\Users\fofoh\Desktop\Automated Machine Learning for Breast Cancer Diagnosis Using TPOT\trained_model.pkl', 'rb'))

@app.route('/')
def home():
    # Renders the HTML form for input (index.html)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data for all 30 features from the form submission
        data = [
            float(request.form['radius_mean']),
            float(request.form['texture_mean']),
            float(request.form['perimeter_mean']),
            float(request.form['area_mean']),
            float(request.form['smoothness_mean']),
            float(request.form['compactness_mean']),
            float(request.form['concavity_mean']),
            float(request.form['concave_points_mean']),
            float(request.form['symmetry_mean']),
            float(request.form['fractal_dimension_mean']),
            float(request.form['radius_se']),
            float(request.form['texture_se']),
            float(request.form['perimeter_se']),
            float(request.form['area_se']),
            float(request.form['smoothness_se']),
            float(request.form['compactness_se']),
            float(request.form['concavity_se']),
            float(request.form['concave_points_se']),
            float(request.form['symmetry_se']),
            float(request.form['fractal_dimension_se']),
            float(request.form['radius_worst']),
            float(request.form['texture_worst']),
            float(request.form['perimeter_worst']),
            float(request.form['area_worst']),
            float(request.form['smoothness_worst']),
            float(request.form['compactness_worst']),
            float(request.form['concavity_worst']),
            float(request.form['concave_points_worst']),
            float(request.form['symmetry_worst']),
            float(request.form['fractal_dimension_worst']),
        ]

        # Convert the data to a DataFrame
        input_data = pd.DataFrame([data], columns=[
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
            'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
            'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
            'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
            'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
            'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
        ])

        # Make a prediction
        output = model.predict(input_data)[0]  # Get the prediction result

        # Map the output to a user-friendly format (assuming 1 is Malignant and 0 is Benign)
        result = 'Malignant' if output == 1 else 'Benign'

        # Render the result page with the prediction result
        return render_template("result.html", result=result)

    except KeyError as e:
        # If there is an issue with the form submission (e.g., missing data), display a basic error message
        return f"Missing or invalid input: {str(e)}", 400

    except Exception as e:
        # Generic error handling (e.g., prediction failure)
        return f"An error occurred: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)

