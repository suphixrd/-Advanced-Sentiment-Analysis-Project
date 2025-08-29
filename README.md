# -Advanced-Sentiment-Analysis-Project
The ultimate goal is to build a robust and accurate sentiment classifier that can handle the complexities of real-world text data.


This repository contains an advanced sentiment analysis project using Python and scikit-learn. The goal is to classify the sentiment of tweets related to airlines as negative, neutral, or positive. This project goes beyond basic classification by implementing comprehensive data preprocessing, detailed model evaluation, and feature importance analysis.

Core Technologies
Python: The primary programming language.

Pandas: Used for data loading and manipulation.

Numpy: Essential for numerical operations.

Matplotlib & Seaborn: Used for creating insightful data visualizations.

Scikit-learn: The core machine learning library for vectorization, model building, and evaluation.

Regular Expressions (re): Used for advanced text cleaning.

Project Workflow
The Advance Sentiment Analysis.py script follows a structured machine learning workflow:

Data Loading and Exploration: The project starts by loading the "Tweets.csv" dataset and performing an initial check on its structure, class distribution, and missing values.

Advanced Data Preprocessing: A custom function is used to clean the text data. This process includes converting text to lowercase, removing URLs, usernames, HTML tags, and non-alphabetic characters. It also removes English stop words and filters out short words to prepare the data for analysis.

Data Visualization: Visualizations are created to understand the dataset better. This includes plots for sentiment distribution, tweet length, and the most frequent words for each sentiment class.

TF-IDF Vectorization: The cleaned text is converted into a numerical format using TfidfVectorizer. This process is enhanced by using n-grams (unigrams and bigrams) and filtering features based on min_df and max_df parameters to improve model performance.

Model Training and Evaluation: Four different machine learning models are trained and evaluated:

Logistic Regression

Multinomial Naive Bayes

Support Vector Machine (LinearSVC)

Random Forest Classifier
The models are trained using a training set and evaluated on a separate test set. Cross-validation is also used to ensure the models' robustness.

Comprehensive Analysis: The project compares the models based on key metrics like accuracy, precision, recall, and F1-score. Confusion matrices and other plots are generated for a deeper understanding of each model's performance.

Best Model Selection and Feature Importance: The model with the highest F1-score is selected as the best performer. For a deeper analysis, the script visualizes the most important features (words) that contribute to positive and negative sentiment predictions.

Sample Predictions: Finally, the best-performing model is used to predict the sentiment of new, unseen text samples, demonstrating its practical application.
