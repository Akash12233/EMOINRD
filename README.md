# Sentiment Analysis Web App

This project is a **Sentiment Analysis Web Application** built using Flask, where users can submit a CSV file or a single text input to predict the sentiment of the text (Positive, Negative, or Neutral). The application uses a machine learning model to classify the sentiment of user input based on pre-trained models, and it also provides a visual representation of the sentiment distribution.

## Authors
- **Akash Tayade** - Main Developer and Contributor
- **Himanshu** - Contributor

## Overview

This project demonstrates the use of sentiment analysis to classify text into one of three categories: Positive, Negative, or Neutral. It also generates a distribution graph to visualize the sentiment breakdown of bulk predictions from a CSV file. The application uses **Flask** as a web framework and integrates machine learning models, preprocessing tools, and visualization components.

### Technologies Used:
- **Flask**: Web framework to create the API.
- **Pandas**: Data manipulation and CSV handling.
- **Scikit-learn**: Machine learning for sentiment prediction.
- **Matplotlib**: Graphing the sentiment distribution.
- **NLTK**: Text preprocessing (tokenization, stemming, stopword removal).
- **Joblib**: Saving and loading models.
- **Base64**: Encoding the graph as a base64 image for embedding in the response.

---

## Features

- **Single Text Prediction**: Users can submit a single text input for sentiment prediction.
- **Bulk Prediction**: Users can upload a CSV file containing sentences, and the system will predict the sentiment for each sentence and return the results in a downloadable CSV.
- **Sentiment Distribution Graph**: A pie chart visualizing the distribution of sentiments in the bulk prediction results.

---

## Requirements

Before running the project, ensure you have the following libraries installed:

```
pip install Flask pandas scikit-learn matplotlib nltk joblib flask-cors
```

Install the requirements file
```
pip install -r requirements.txt
```

Run the app
```
flask --app api.py run
```

## How to Train the Model
The model is created using the Data Exploration and Modelling.ipynb notebook. The steps include:

- Data Exploration: Analyze and prepare the dataset.
- Modeling: Build and train a Random Forest model for sentiment analysis.
- Serialization: Save the trained model using joblib for later use in the web app.
- To retrain the model, open the notebook and follow the steps.

## API Endpoints

### `/test` (GET)

- **Description**: A simple test endpoint to confirm that the API is running.
- **Response**: 
  - Message: `"Test request received successfully. Service is running."`

### `/predict` (POST)

- **Description**: Main endpoint to perform sentiment prediction on text input or a CSV file.

#### Request:
- **Single Text Prediction**:
  - Send a JSON payload with a `"text"` key containing the text you want to analyze.
  - Example:
    ```json
    {
      "text": "I love this product!"
    }
    ```

- **Bulk Prediction (CSV file)**:
  - Upload a CSV file using the key `"file"`, where the file should contain a column named `Sentence` with the text data for analysis.

#### Response:
- **Single Text Prediction**:
  - Returns a JSON object with the predicted sentiment (`Positive`, `Negative`, or `Neutral`).
  - Example:
    ```json
    {
      "prediction": "Positive"
    }
    ```

- **Bulk Prediction (CSV file)**:
  - Returns a CSV file with an additional column `Predicted sentiment`, containing the predicted sentiment for each row.
  - The response also includes a base64-encoded pie chart image representing the sentiment distribution across the dataset.
  

