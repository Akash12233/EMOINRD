from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64
from flask_cors import CORS
from joblib import load

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})  # Allows all origins for the /predict route

STOPWORDS = set(stopwords.words("english"))

@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Loading models and scalers using joblib
    predictor = load(r"D:\ML\Sentiment-Analysis-main\Sentiment-Analysis-main\Models\Model_rf.joblib")
    scaler = load(r"D:\ML\Sentiment-Analysis-main\Sentiment-Analysis-main\Models\scaler.joblib")
    cv = load(r"D:\ML\Sentiment-Analysis-main\Sentiment-Analysis-main\Models\countVectorizer.joblib")

    
    try:
        # Check if the request contains a file (for bulk prediction) or text input
        
        if "file" in request.files:
            # Bulk prediction from CSV file
            file = request.files["file"]
            data = pd.read_csv(file)

            predictions, graph_base64 = bulk_prediction(predictor, scaler, cv, data)

            # Check that predictions was successfully created
            if predictions is None:
                print("Error: Predictions file generation failed.")
                return "Failed to generate predictions.", 500

            # Prepare the response for download
            response = send_file(
                predictions,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )

            # Add custom headers for the graph
            response.headers["X-Graph-Exists"] = "true"
            response.headers["X-Graph-Data"] = graph_base64

            return response
        elif "text" in request.json:
            # Single string prediction
            text_input = request.json["text"]
            predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)
            print(f"Predicted sentiment: {predicted_sentiment}");
            return jsonify({"prediction": predicted_sentiment})

    except Exception as e:
        return jsonify({"error": str(e)})

def single_prediction(predictor, scaler, cv, text_input):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)[0]

    if y_predictions == 1:
        return "Positive"
    elif y_predictions == 0:
        return "Negative"
    else:
        return "Neutral"
def bulk_prediction(predictor, scaler, cv, data):
    corpus = []
    stemmer = PorterStemmer()
    for i in range(0, data.shape[0]):
        review = re.sub("[^a-zA-Z]", " ", data.iloc[i]["Sentence"])
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
        review = " ".join(review)
        corpus.append(review)

    print("Sample preprocessed corpus:", corpus[:5])  # Debugging statement

    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)
    y_predictions = list(map(sentiment_mapping, y_predictions))

    data["Predicted sentiment"] = y_predictions
    print("Sample predictions:", y_predictions[:5])  # Debugging statement

    predictions_csv = BytesIO()
    print("1")
    data.to_csv(predictions_csv, index=False, encoding='utf-8')
    print("2")
    predictions_csv.seek(0)
    print("3")
    graph_base64 = get_distribution_graph(data)
    
    return predictions_csv, graph_base64

def get_distribution_graph(data):
    print("Entering get_distribution_graph")  # Debugging statement
    
    fig = plt.figure(figsize=(5, 5))
    colors = ("green", "red", "blue")  # Added third color for Neutral
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Predicted sentiment"].value_counts()

    if tags.empty:
        print("No data to plot in tags.")  # Debugging statement
        return ""

    # Print tag values to ensure they are correct
    print("Tags value counts:", tags)  # Debugging statement

    # Basic pie chart without explode to test
    try:
        tags.plot(
            kind="pie",
            autopct="%1.1f%%",
            shadow=True,
            colors=colors[:len(tags)],  # Ensuring colors match tag count
            startangle=90,
            wedgeprops=wp,
        )
        plt.title("Sentiment Distribution")
        plt.xlabel("")
        plt.ylabel("")

        graph = BytesIO()
        plt.savefig(graph, format="png")
        plt.close()

        graph.seek(0)
        graph_base64 = base64.b64encode(graph.read()).decode('utf-8')
        print("Graph generated, base64 length:", len(graph_base64))  # Debugging statement

        return graph_base64
    except Exception as e:
        print(f"Error while plotting: {e}")
        return ""



def sentiment_mapping(x):
    if x == 1:
        return "Positive"
    elif x == 0:
        return "Negative"
    else:
        return "Neutral"


if __name__ == "__main__":
    app.run(debug=True)
