PURPOSE
Classify pasted news as Fake or Real. Pick a model from a dropdown. Use a shared TF-IDF vectorizer.

KEY FEATURES
• Paste headline and article text.
• Choose a model from a dropdown.
• See a clear Fake or Real result.
• Open the Why panel to view top contributing words, when available.

MODELS
• SGDClassifier → sgd_classifier_model.joblib
• Logisticregression → logisticregresionl.joblib
• Complement Naive Bayes → complement_nb_model.joblib
• SVC → svm_model.joblib
• Shared vectorizer → tfidf_vectorizer.joblib

FILES
• app.py
• tfidf_vectorizer.joblib
• sgd_classifier_model.joblib
• logisticregresionl.joblib
• complement_nb_model.joblib
• svm_model.joblib

INSTALL
pip install streamlit scikit-learn nltk joblib numpy
python -m nltk.downloader stopwords wordnet omw-1.4

RUN
streamlit run app.py

USAGE

Place all joblib files in the same folder as app.py.

Start the app.

Select a model from the dropdown.

Paste a headline.

Paste article text.

Click Classify.

Read the result. Fake or Real.

Open Why for word contributions, when supported by the model.

PREPROCESSING
• Keep letters, digits, and spaces.
• Convert to lowercase.
• Remove English stopwords.
• Lemmatize tokens.
• Combine cleaned title and text, then vectorize with TF-IDF.

EXAMPLE
Input title, Government report confirms new safety rules.
Input text, The ministry released documents and sources for public review.
Output, Real.

TIPS
• Keep preprocessing in training and inference aligned.
• Use the same TF-IDF vectorizer file for every model.
• Provide both title and body for stronger signals.

SOCIAL MEDIA POSTS
• Paste news. Pick a model. Get Fake or Real, fast.
• Works with SGD, Logistic Regression, Complement NB, and SVC.
• Uses TF-IDF features and simple preprocessing.
• See top words that influenced the call.