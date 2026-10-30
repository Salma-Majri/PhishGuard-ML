"""Lab 04 - Phish or Legit? (Streamlit UI)"""
import streamlit as st
import pandas as pd
from src.pipeline import load_data, split, make_pipeline, contributions

st.set_page_config(page_title="Phish or Legit?", layout="centered")

st.title("🛡️ Phish or Legit?")
st.caption("TF-IDF + Logistic Regression / Naive Bayes text classification pipeline.")

@st.cache_resource
def get_models():
    df = load_data()
    Xtr, _, ytr, _ = split(df)
    pipe_lr = make_pipeline("lr").fit(Xtr, ytr)
    pipe_nb = make_pipeline("nb").fit(Xtr, ytr)
    return {"Logistic Regression": pipe_lr, "Naive Bayes": pipe_nb}

with st.spinner("Loading and training models..."):
    models = get_models()

# Classifier selector
model_choice = st.selectbox("Choose classifier:", list(models.keys()))
pipe = models[model_choice]

# Example messages dictionary including deceptive & misclassified cases
examples = {
    "Phish 1 (Clear Phish)": "Northbank: your account has been locked. Verify now http://bit.ly/ab12cd",
    "Phish 2 (Deceptive / False Negative)": "Sorry I missed your delivery today. Please check your tracking details and reschedule your drop-off time on our support portal.",
    "Legit 1 (Clear Legit)": "Hi, are we still on for coffee at 7:30?",
    "Legit 2 (Tricky / False Positive)": "This is ur face test ( 1 2 3 4 5 6 7 8 9 ) select any number i will tell ur face astrology.... am waiting. quick reply..."
}

# Initialize session state for text input using the correct key
if "msg_text" not in st.session_state:
    st.session_state.msg_text = examples["Phish 1 (Clear Phish)"]

# Example selector dropdown
selected_ex = st.selectbox("Select an example message:", ["--- Custom message ---"] + list(examples.keys()))

if selected_ex != "--- Custom message ---":
    st.session_state.msg_text = examples[selected_ex]

# Text input for message
text = st.text_area("Paste a message to classify:", value=st.session_state.msg_text, height=100)

# Decision threshold slider
threshold = st.slider("Decision threshold for phishing", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

if st.button("Classify Message", type="primary"):
    if not text.strip():
        st.error("Please enter some text to classify.")
    else:
        try:
            score = float(pipe.predict_proba([text])[0][1])
        except Exception as err:
            st.error(f"Error during prediction: {err}")
        else:
            label = "phish" if score >= threshold else "legit"
            
            # Display prediction result
            if label == "phish":
                st.error(f"🚨 Prediction: PHISHING-style text | Score: {score:.2f} (Threshold: {threshold})")
            else:
                st.success(f"✅ Prediction: LEGIT | Score for phish: {score:.2f} (Threshold: {threshold})")

            # Per-feature explanation for Logistic Regression
            if model_choice == "Logistic Regression":
                contribs = contributions(pipe, text)
                
                increased = {k: v for k, v in contribs.items() if v > 0}
                decreased = {k: v for k, v in contribs.items() if v < 0}
                
                st.subheader("Per-Feature Explanation (Log-Odds Contributions)")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Features pushing toward **PHISH**:")
                    if increased:
                        st.table(pd.DataFrame(list(increased.items()), columns=["Feature", "Contribution"]).head(5))
                    else:
                        st.info("None")
                with col2:
                    st.write("Features pushing toward **LEGIT**:")
                    if decreased:
                        st.table(pd.DataFrame(list(decreased.items()), columns=["Feature", "Contribution"]).head(5))
                    else:
                        st.info("None")
            else:
                st.info("Feature contributions (log-odds) are available exclusively for Logistic Regression.")

# Mandatory UI notice
st.markdown("---")
st.warning("⚠️ **Notice:** This tool supports a human decision and does not block messages by itself.")