import os
from flask import Flask, request
from main_knn import predict, calculate_thresholds

# Load the dataset and calculate thresholds
calculate_thresholds("dataset.csv")

app = Flask(__name__)

@app.route("/")
def main():
    return 'Post JSON to /predicts'

@app.route('/predicts', methods=['POST'])
def predicts():
    if request.method == "POST":

        # Get input data from JSON request
        data = request.get_json()
        input_data = data["input"]

        # Make prediction
        prediction = predict(input_data)
        percentage1, percentage2 = 0, 0

        return {"predict": prediction, "percentage1": percentage1, "percentage2": percentage2}
    else:
        return {"error": "Only POST requests are allowed"}, 405

if __name__ == '__main__':
    # Use environment variable to specify the port number
    port = int(os.environ.get('PORT', 5000))
    # Use environment variable to specify whether to run in production mode
    is_production = os.environ.get('FLASK_ENV') == 'production'
    # Start the application
    app.run(host='0.0.0.0', port=port, debug=False)