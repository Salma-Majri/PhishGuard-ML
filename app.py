"""Lab 04 - Phish or Legit? (Streamlit UI - Clean & Fixed Version)"""
import streamlit as st
import pandas as pd
from src.pipeline import load_data, split, make_pipeline, contributions

st.set_page_config(
    page_title="PhishGuard | Phish or Legit?",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ PhishGuard: Phish or Legit?")
st.caption("Advanced NLP Text Classification Pipeline powered by SMS Dataset & Logistic Regression.")

@st.cache_resource
def get_models():
    df = load_data()
    Xtr, _, ytr, _ = split(df)
    pipe_lr = make_pipeline("lr", analyzer="char_wb").fit(Xtr, ytr)
    pipe_nb = make_pipeline("nb", analyzer="word").fit(Xtr, ytr)
    return {"Logistic Regression (Char N-Grams)": pipe_lr, "Naive Bayes": pipe_nb}

with st.spinner(" Loading dataset from GitHub and training models..."):
    models = get_models()

col_top1, col_top2 = st.columns([1, 1])
with col_top1:
    model_choice = st.selectbox(" Choose Classifier:", list(models.keys()))
pipe = models[model_choice]

with col_top2:
    threshold = st.slider(" Decision Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

examples = {
    "Phish 1 (Clear Phish)": "Free entry in 2 a wkly comp to win FA Cup final tickets 21st May 2005. Text FA to 87121",
    "Phish 2 (Deceptive SMS)": "URGENT! Your mobile no. was awarded a £2000 bonus caller. Prize code L96. Call 09066361911",
    "Legit 1 (Clear Legit)": "Hi, are we still on for coffee at 7:30?",
    "Legit 2 (Tricky Legit)": "Ok lar... Joking wif u oni..."
}

# Session state initialization
if "msg_text" not in st.session_state:
    st.session_state.msg_text = examples["Phish 1 (Clear Phish)"]

def update_text():
    selected = st.session_state.ex_selector
    if selected != "--- Custom message ---":
        st.session_state.msg_text = examples[selected]

selected_ex = st.selectbox(
    "📂 Select a benchmark example message:", 
    ["--- Custom message ---"] + list(examples.keys()),
    key="ex_selector",
    on_change=update_text
)

text = st.text_area("✍️ Paste or type a message to evaluate:", key="msg_text", height=120)

if st.button("🔍 Classify Message", type="primary"):
    if not text.strip():
        st.error("⚠️ Please enter some text to classify.")
    else:
        try:
            score = float(pipe.predict_proba([text])[0][1])
        except Exception as err:
            st.error(f"❌ Error: {err}")
        else:
            label = "phish" if score >= threshold else "legit"
            
            st.markdown("### 📊 Prediction Result")
            if label == "phish":
                st.error(f"🚨 **PHISHING / SPAM DETECTED** | Score: **{score:.2f}** (Threshold: `{threshold}`)")
            else:
                st.success(f"✅ **LEGITIMATE MESSAGE** | Score: **{score:.2f}** (Threshold: `{threshold}`)")

            if "Logistic Regression" in model_choice:
                contribs = contributions(pipe, text)
                increased = {k: v for k, v in contribs.items() if v > 0}
                decreased = {k: v for k, v in contribs.items() if v < 0}
                
                st.markdown("---")
                st.subheader(" Explainable AI: Log-Odds Feature Contributions")
                col_feat1, col_feat2 = st.columns(2)
                with col_feat1:
                    st.markdown("🔴 **Pushing toward SPAM/PHISH:**")
                    if increased:
                        st.table(pd.DataFrame(list(increased.items()), columns=["Feature", "Contribution"]).head(5))
                    else:
                        st.info("None")
                with col_feat2:
                    st.markdown("🟢 **Pushing toward LEGIT:**")
                    if decreased:
                        st.table(pd.DataFrame(list(decreased.items()), columns=["Feature", "Contribution"]).head(5))
                    else:
                        st.info("None")

st.markdown("---")
st.warning("⚠️ **Notice:** This AI tool acts as a decision support system.")