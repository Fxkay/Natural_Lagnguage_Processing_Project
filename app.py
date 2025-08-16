# app.py
import re
import nltk
import streamlit as st
import numpy as np
from joblib import load
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Needed so sklearn can deserialize saved models/vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import SVC, LinearSVC

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Fake-vs-Real News",
                   page_icon="📰", layout="centered")
st.title("📰 Fake-vs-Real News")
st.caption("Paste a headline/article, pick a model, and get a Fake/Real verdict.")

# ----------------------------
# NLTK setup (robust)
# ----------------------------
lemmatizer = WordNetLemmatizer()


def safe_download(pkg: str) -> bool:
    try:
        nltk.download(pkg, quiet=True)
        return True
    except Exception:
        return False


def load_stopwords():
    if safe_download("stopwords"):
        try:
            return set(stopwords.words("english"))
        except Exception:
            pass
    return set()


safe_download("wordnet")
safe_download("omw-1.4")
STOP_WORDS = load_stopwords()


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.lower()
    tokens = [w for w in text.split() if w not in STOP_WORDS]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# ----------------------------
# Load artifacts
# ----------------------------


@st.cache_resource(show_spinner=True)
def load_vectorizer():
    return load("tfidf_vectorizer.joblib")


@st.cache_resource(show_spinner=True)
def load_model(path: str):
    return load(path)


try:
    VECTORIZER = load_vectorizer()
except Exception as e:
    st.error("❌ Could not load `tfidf_vectorizer.joblib`. Place it beside `app.py`.")
    st.exception(e)
    st.stop()

# ----------------------------
# Dropdown mapping (exact labels → files)
# ----------------------------
MODEL_CHOICES = {
    "SGDClassifier": "sgd_classifier_model.joblib",
    "Logisticregression": "logisticregresionl.joblib",
    "Complement Naive Bayes": "complement_nb_model.joblib",
    "SVC": "svm_model.joblib",
}
model_name = st.selectbox("Choose a model", list(MODEL_CHOICES.keys()))
model_path = MODEL_CHOICES[model_name]

try:
    MODEL = load_model(model_path)
except Exception as e:
    st.error(f"❌ Could not load `{model_path}` for **{model_name}**.")
    st.exception(e)
    st.stop()

# ----------------------------
# UI
# ----------------------------
with st.form("news_form"):
    title = st.text_input("Headline (optional)")
    body = st.text_area("Article text (paste here)",
                        height=260, placeholder="Paste the full story...")
    submitted = st.form_submit_button("Classify")

# ----------------------------
# Helpers
# ----------------------------


def predict_label(model, X_vec):
    pred = int(model.predict(X_vec)[0])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_vec)[0]
        return pred, float(max(proba))
    if hasattr(model, "decision_function"):
        return pred, float(model.decision_function(X_vec)[0])
    return pred, None


def top_contributors(model, vectorizer, X_vec, top_k=10):
    """
    contribution ≈ tfidf_value * feature_weight
    - Linear models (SGD/LogReg/LinearSVC): use coef_
    - ComplementNB: use log odds (feature_log_prob_[1] - feature_log_prob_[0])
    - Kernel SVC usually has no coef_ → skip
    """
    try:
        feature_names = np.array(vectorizer.get_feature_names_out())
        x = X_vec.tocoo()

        if hasattr(model, "coef_"):
            weights = model.coef_.ravel()
        elif hasattr(model, "feature_log_prob_") and model.feature_log_prob_.shape[0] >= 2:
            weights = model.feature_log_prob_[1] - model.feature_log_prob_[0]
        else:
            return [], []

        contrib = [(feature_names[j], v * weights[j])
                   for _, j, v in zip(x.row, x.col, x.data)]
        if not contrib:
            return [], []
        contrib.sort(key=lambda t: t[1], reverse=True)
        pos = contrib[:top_k]
        neg = sorted(contrib, key=lambda t: t[1])[:top_k]
        return pos, neg
    except Exception:
        return [], []

# ----------------------------
# Classify
# ----------------------------


def classify(title_text: str, body_text: str):
    clean_title = preprocess_text(title_text or "")
    clean_body = preprocess_text(body_text or "")
    combined = (clean_title + " " + clean_body).strip()

    if not combined:
        st.warning("Please provide at least a headline or article text.")
        return

    X = VECTORIZER.transform([combined])
    pred_num, score = predict_label(MODEL, X)

    label_text = "Real" if pred_num == 1 else "Fake"
    icon = "✅" if pred_num == 1 else "⚠️"
    st.subheader(f"{icon} Prediction: **{label_text}**")
    st.caption(f"Model: {model_name} • Features: TF-IDF (unigrams+bigrams)")

    with st.expander("Why? (top weighted words from this text)"):
        pos, neg = top_contributors(MODEL, VECTORIZER, X)
        if pos:
            st.write("**Top words pushing toward Real:**")
            st.write(", ".join([f"{w} ({c:.3f})" for w, c in pos]))
        if neg:
            st.write("**Top words pushing toward Fake:**")
            st.write(", ".join([f"{w} ({c:.3f})" for w, c in neg]))
        if not pos and not neg:
            st.write(
                "Explanation unavailable for this model or no informative tokens found.")

    with st.expander("Model details"):
        st.json({
            "selected_model": model_name,
            "artifact": model_path,
            "has_probabilities": hasattr(MODEL, "predict_proba"),
            "score": score,
            "vectorizer": "TfidfVectorizer",
            "ngram_range": getattr(VECTORIZER, "ngram_range", None),
            "max_features": getattr(VECTORIZER, "max_features", None),
        })


if submitted:
    classify(title, body)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Files needed (same folder)")
    st.markdown(
        """
- `tfidf_vectorizer.joblib`
- `sgd_classifier_model.joblib`
- `logisticregresionl.joblib`
- `complement_nb_model.joblib`
- `svm_model.joblib`
        """
    )
    st.markdown("**Output:** shows only **Fake** or **Real**.")
