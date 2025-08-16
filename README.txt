README

Project: Natural language Processing for fake and Real news

Goal
Classify news articles as real (1) or fake (0).

Datasets

data.csv for training and testing

validation_data.csv for predictions

Columns

label: 0 fake, 1 real

title: headline

text: article content

subject: category

date: publication date

Steps

Import libraries: pandas, numpy, sklearn, nltk, seaborn

Load datasets

Clean text: lowercase, remove symbols, remove stopwords, lemmatize

Vectorize text with TF-IDF

Train models: Logistic Regression, SVM, Naive Bayes, Random Forest, SGDClassifier

Evaluate with accuracy, precision, recall, f1 score, confusion matrix

Save trained models with joblib

Predict on validation_data.csv

Replace label 2 with predicted 0 or 1

Export predictions in original format

Results

SVM accuracy about 99.5 percent

Logistic Regression and SGDClassifier accuracy above 98 percent

Naive Bayes and Random Forest lower than others

How to run

Install requirements: pip install pandas numpy scikit-learn nltk seaborn

Place data.csv and validation_data.csv in project folder

Run notebook step by step

Saved model example: svm_model.joblib

Predictions exported as CSV file

Key points

Clean text before training

TF-IDF used because of context
Swap models easily for testing

Output is ready for use in production