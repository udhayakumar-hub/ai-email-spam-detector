"""
AI Email Spam Detector - Streamlit Web Application
Application 3 of NLP Project Series.
"""

import sys
import os
import datetime

# Ensure project root is always in Python module search path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Optional seaborn with native matplotlib fallback
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from src.model import SpamClassifier
from src.preprocessing import preprocess_text
from src.utils import SAMPLE_MESSAGES, compute_text_stats, generate_explanation

# Configure Streamlit page layout and title
st.set_page_config(
    page_title="AI Email Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI design
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }
    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-top: 6px;
        margin-bottom: 0;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 9999px;
        background: #3b82f6;
        color: white;
        margin-top: 8px;
    }
    /* Result Cards */
    .result-card-spam {
        background: rgba(239, 68, 68, 0.12);
        border: 2px solid #ef4444;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    .result-card-ham {
        background: rgba(16, 185, 129, 0.12);
        border: 2px solid #10b981;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    .stat-badge {
        font-size: 0.9rem;
        padding: 6px 12px;
        border-radius: 6px;
        background: #f1f5f9;
        color: #1e293b;
        display: inline-block;
        margin: 4px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []

if "input_message" not in st.session_state:
    st.session_state.input_message = ""

if "selected_algo" not in st.session_state:
    st.session_state.selected_algo = "naive_bayes"


@st.cache_resource(show_spinner="Training NLP classification model...")
def get_classifier(algorithm: str = "naive_bayes") -> SpamClassifier:
    """
    Cached function to train and load the spam classifier pipeline.
    """
    dataset_path = os.path.join(PROJECT_ROOT, "data", "spam_dataset.csv")
    classifier = SpamClassifier(algorithm=algorithm)
    classifier.train(dataset_path)
    return classifier


# Sidebar Settings & Information
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("⚙️ Model Settings")
    
    algo_choice = st.selectbox(
        "Classifier Algorithm",
        options=["Multinomial Naive Bayes", "Logistic Regression"],
        index=0 if st.session_state.selected_algo == "naive_bayes" else 1,
        help="Choose the machine learning algorithm used for classification."
    )
    
    selected_key = "naive_bayes" if algo_choice == "Multinomial Naive Bayes" else "logistic_regression"
    if selected_key != st.session_state.selected_algo:
        st.session_state.selected_algo = selected_key
        st.rerun()

    st.markdown("---")
    st.subheader("📊 Model Status")
    try:
        classifier = get_classifier(st.session_state.selected_algo)
        metrics = classifier.metrics
        st.success(f"✅ Model Ready ({metrics['algorithm']})")
        st.markdown(f"**Test Accuracy:** `{metrics['accuracy'] * 100:.1f}%`")
        st.markdown(f"**F1-Score:** `{metrics['f1_score'] * 100:.1f}%`")
        st.markdown(f"**Training Samples:** `{metrics['total_samples']}`")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

    st.markdown("---")
    st.subheader("💡 NLP Pipeline")
    st.markdown("""
    1. **Text Preprocessing** (Cleaning, Stopwords, Stemming)
    2. **TF-IDF Vectorization** (Unigram & Bigrams)
    3. **Probabilistic Classification** (Naive Bayes / LogReg)
    4. **Inference & Risk Scoring**
    """)
    
    st.markdown("---")
    st.caption("AI Email Spam Detector • NLP Project 3 • local & offline")


# Hero Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">
        <span>🛡️ AI Email Spam Detector</span>
    </div>
    <div class="hero-subtitle">
        Real-time Natural Language Processing & Machine Learning Classification for Emails and SMS Messages
    </div>
    <span class="badge-pill">NLP Project Series #3</span>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_classify, tab_eval, tab_dashboard, tab_info = st.tabs([
    "🔍 Analyze Message",
    "📈 Model Evaluation",
    "📊 Session Dashboard",
    "📚 How It Works & Theory"
])


# ==========================================
# TAB 1: MESSAGE CLASSIFIER
# ==========================================
with tab_classify:
    st.subheader("📝 Input Email or Message")
    st.markdown("Paste any email or SMS text below, or choose one of the pre-configured sample messages to test the pipeline.")

    # Preset Example Selector
    col_preset, col_clear = st.columns([4, 1])
    with col_preset:
        preset_choice = st.selectbox(
            "Load a Preset Sample Message:",
            options=["-- Select a sample message to populate input --"] + list(SAMPLE_MESSAGES.keys()),
            key="preset_selector"
        )
    with col_clear:
        st.write("") # spacing
        st.write("")
        if st.button("🗑️ Clear Input", use_container_width=True):
            st.session_state.input_message = ""
            st.rerun()

    if preset_choice and preset_choice != "-- Select a sample message to populate input --":
        st.session_state.input_message = SAMPLE_MESSAGES[preset_choice]

    # Message Text Area
    user_input = st.text_area(
        "Message Content:",
        value=st.session_state.input_message,
        placeholder="e.g., Congratulations! You have won $1,000 cash. Click here to claim...",
        height=140,
        help="Type or paste the email subject and body."
    )

    # Analyze Button
    col_btn, col_info = st.columns([2, 3])
    with col_btn:
        analyze_clicked = st.button("🚀 Analyze Message", type="primary", use_container_width=True)

    if analyze_clicked or (user_input.strip() and user_input != st.session_state.get("last_analyzed_input", "")):
        if not user_input.strip():
            st.warning("⚠️ Please enter or paste some text before clicking Analyze.")
        else:
            with st.spinner("Processing text and running NLP inference..."):
                try:
                    classifier = get_classifier(st.session_state.selected_algo)
                    result = classifier.predict(user_input)
                    stats = compute_text_stats(user_input)
                    reasons = generate_explanation(result["prediction"], result["confidence"], stats, result["top_terms"])

                    st.session_state.last_analyzed_input = user_input

                    # Save to session history (avoid duplicate consecutive runs)
                    history_entry = {
                        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                        "text_snippet": user_input[:80] + ("..." if len(user_input) > 80 else ""),
                        "prediction": result["prediction"],
                        "confidence": f"{result['confidence']:.1f}%",
                        "spam_prob": result["spam_prob"],
                        "ham_prob": result["ham_prob"],
                        "words": stats["total_words"]
                    }
                    st.session_state.history.insert(0, history_entry)

                    st.markdown("---")
                    st.subheader("🎯 Prediction Results")

                    # Display Result Card
                    is_spam = result["prediction"] == "SPAM"
                    card_class = "result-card-spam" if is_spam else "result-card-ham"
                    badge_color = "#ef4444" if is_spam else "#10b981"
                    badge_icon = "🚨" if is_spam else "✅"

                    st.markdown(f"""
                    <div class="{card_class}">
                        <h2 style="margin:0; color:{badge_color}; display:flex; align-items:center; gap:8px;">
                            <span>{badge_icon} Classified as: {result['prediction']}</span>
                        </h2>
                        <p style="margin:6px 0 0 0; font-size:1.1rem;">
                            Confidence Level: <strong>{result['confidence']:.1f}%</strong> | Model: <em>{classifier.metrics['algorithm']}</em>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Probabilities Breakdown
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        st.markdown(f"**Spam Probability:** `{result['spam_prob']}%`")
                        st.progress(result["spam_prob"] / 100.0)
                    with col_p2:
                        st.markdown(f"**Ham (Not Spam) Probability:** `{result['ham_prob']}%`")
                        st.progress(result["ham_prob"] / 100.0)

                    # Text Statistics Row
                    st.markdown("##### 📊 Linguistic & Structural Indicators")
                    stat_cols = st.columns(5)
                    stat_cols[0].metric("Word Count", stats["total_words"])
                    stat_cols[1].metric("Character Count", stats["total_chars"])
                    stat_cols[2].metric("Uppercase Ratio", f"{stats['uppercase_percentage']}%")
                    stat_cols[3].metric("Detected Links", stats["urls_count"])
                    stat_cols[4].metric("Currency Mentions", stats["currency_count"])

                    # Decision Explanation
                    st.markdown("##### 🔍 Why this prediction was made:")
                    for reason in reasons:
                        st.markdown(f"- {reason}")

                    # Detailed Token Inspector Expandable
                    with st.expander("🛠️ View Internal NLP Transformation (Tokens & TF-IDF Features)"):
                        st.markdown(f"**Cleaned & Stemmed Input:** `{result['processed_text']}`")
                        if result["top_terms"]:
                            st.markdown(f"**Top TF-IDF Active Features:** `{', '.join(result['top_terms'])}`")
                        else:
                            st.info("No high-weight vocabulary tokens matched specific model n-grams.")

                except Exception as ex:
                    st.error(f"❌ An error occurred during classification: {str(ex)}")


# ==========================================
# TAB 2: MODEL EVALUATION
# ==========================================
with tab_eval:
    st.subheader("📈 Machine Learning Performance & Metrics")
    st.markdown("""
    The classification model is trained on a stratified train/test split. Below are the performance metrics
    computed on the unseen test dataset.
    """)

    try:
        classifier = get_classifier(st.session_state.selected_algo)
        m = classifier.metrics

        # Metrics Row
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Accuracy", f"{m['accuracy'] * 100:.1f}%", help="Proportion of total correct predictions")
        m_col2.metric("Precision", f"{m['precision'] * 100:.1f}%", help="How many detected spam messages were truly spam")
        m_col3.metric("Recall", f"{m['recall'] * 100:.1f}%", help="How many actual spam messages were correctly detected")
        m_col4.metric("F1-Score", f"{m['f1_score'] * 100:.1f}%", help="Harmonic mean of precision and recall")

        st.markdown("---")

        col_cm, col_dist = st.columns([1, 1])

        # Confusion Matrix Plot
        with col_cm:
            st.markdown("##### 🎯 Confusion Matrix")
            cm = np.array(m["confusion_matrix"])
            
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            if HAS_SEABORN:
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=["Ham (0)", "Spam (1)"],
                    yticklabels=["Ham (0)", "Spam (1)"],
                    cbar=False,
                    ax=ax,
                    annot_kws={"size": 14, "weight": "bold"}
                )
            else:
                cax = ax.matshow(cm, cmap="Blues")
                fig.colorbar(cax)
                for (i, j), val in np.ndenumerate(cm):
                    ax.text(j, i, f'{val}', ha='center', va='center', color='black', fontsize=14, weight='bold')
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(["Ham (0)", "Spam (1)"])
                ax.set_yticklabels(["Ham (0)", "Spam (1)"])

            ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
            ax.set_ylabel("True Label", fontsize=10, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Dataset Distribution Plot
        with col_dist:
            st.markdown("##### 📦 Dataset Class Distribution")
            df = classifier.data_df
            if df is not None:
                counts = df["label"].value_counts()
                fig2, ax2 = plt.subplots(figsize=(4.5, 3.5))
                colors = ["#10b981", "#ef4444"]
                bars = ax2.bar(["Ham (Not Spam)", "Spam"], [counts.get("ham", 0), counts.get("spam", 0)], color=colors)
                ax2.set_ylabel("Sample Count", fontsize=10, fontweight="bold")
                for bar in bars:
                    yval = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1, int(yval), ha='center', va='bottom', fontweight='bold')
                ax2.set_ylim(0, max(counts) * 1.2)
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()

        # Metrics Definition Explanations
        with st.expander("📖 Understanding Evaluation Metrics"):
            st.markdown("""
            - **Accuracy**: Overall fraction of predictions that are correct.
            - **Precision (Spam)**: Out of all emails flagged as SPAM, how many were actually spam? (Crucial to avoid filtering important legitimate emails).
            - **Recall (Spam)**: Out of all real spam emails, how many did the system catch?
            - **F1-Score**: The balanced harmonic mean between Precision and Recall.
            - **Confusion Matrix**:
                - **True Negative (Top-Left)**: Legitimate emails correctly marked as NOT SPAM.
                - **False Positive (Top-Right)**: Legitimate emails wrongly marked as SPAM.
                - **False Negative (Bottom-Left)**: Spam emails missed and marked as NOT SPAM.
                - **True Positive (Bottom-Right)**: Spam emails correctly identified.
            """)

    except Exception as e:
        st.error(f"Error computing evaluation metrics: {str(e)}")


# ==========================================
# TAB 3: DASHBOARD & HISTORY
# ==========================================
with tab_dashboard:
    st.subheader("📊 Session Analytics & Prediction History")
    
    if not st.session_state.history:
        st.info("ℹ️ No predictions recorded yet in this session. Analyze some messages in the 'Analyze Message' tab to populate this dashboard.")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        total_preds = len(history_df)
        spam_preds = (history_df["prediction"] == "SPAM").sum()
        ham_preds = (history_df["prediction"] == "NOT SPAM").sum()
        spam_pct = (spam_preds / total_preds) * 100 if total_preds > 0 else 0.0

        # Summary KPIs
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Analyzed", total_preds)
        kpi2.metric("Spam Detected", spam_preds, delta=f"{spam_pct:.1f}%")
        kpi3.metric("Ham (Not Spam)", ham_preds)
        kpi4.metric("Spam Rate", f"{spam_pct:.1f}%")

        st.markdown("---")

        col_hist_chart, col_hist_table = st.columns([1, 2])

        with col_hist_chart:
            st.markdown("##### 🥧 Session Ratio")
            fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
            labels = []
            sizes = []
            colors = []
            if spam_preds > 0:
                labels.append("Spam")
                sizes.append(spam_preds)
                colors.append("#ef4444")
            if ham_preds > 0:
                labels.append("Not Spam")
                sizes.append(ham_preds)
                colors.append("#10b981")

            ax_pie.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140, textprops={'weight': 'bold'})
            plt.tight_layout()
            st.pyplot(fig_pie)
            plt.close()

        with col_hist_table:
            st.markdown("##### 📜 Recent Analyses (Session History)")
            st.dataframe(
                history_df[["timestamp", "text_snippet", "prediction", "confidence", "words"]],
                use_container_width=True,
                hide_index=True
            )
            
            if st.button("🗑️ Clear Session History"):
                st.session_state.history = []
                st.rerun()


# ==========================================
# TAB 4: HOW IT WORKS & THEORY
# ==========================================
with tab_info:
    st.subheader("📚 NLP Architecture & Educational Guide")
    st.markdown("""
    This project demonstrates the classic and battle-tested **Machine Learning NLP workflow** for text classification.
    """)

    st.markdown("### 1️⃣ Text Preprocessing")
    st.markdown("""
    Before machine learning algorithms can interpret human text, it must be cleaned and normalized:
    - **Lowercasing**: Unifies case sensitivity (`"URGENT"` $\\rightarrow$ `"urgent"`).
    - **Entity Marker Replacement**: Replaces dynamic items like URLs, emails, phone numbers, and currency symbols with structured tokens (`urlmarker`, `moneymarker`).
    - **Punctuation Stripping**: Removes exclamation marks, asterisks, and special characters.
    - **Stopword Removal**: Filters out frequent non-discriminative English words (`the`, `is`, `at`, `which`).
    - **Stemming**: Reduces inflected words to their root word stem (e.g., `"winning"` $\\rightarrow$ `"win"`).
    """)

    st.markdown("---")
    st.markdown("### 2️⃣ TF-IDF Vectorization")
    st.markdown("""
    Computes numerical vector representations reflecting how important a word is to a document in a corpus:
    $$\\text{TF-IDF}(t, d, D) = \\text{TF}(t, d) \\times \\text{IDF}(t, D)$$
    - **Term Frequency (TF)**: Frequency of token $t$ in document $d$.
    - **Inverse Document Frequency (IDF)**: $\\log\\left(\\frac{1 + N}{1 + \\text{DF}(t)}\\right) + 1$, downweighting terms appearing in almost all messages.
    """)

    st.markdown("---")
    st.markdown("### 3️⃣ Multinomial Naive Bayes Classification")
    st.markdown("""
    Based on Bayes' Theorem with the assumption of conditional independence among features:
    $$P(C \\mid X) \\propto P(C) \\prod_{i=1}^{n} P(x_i \\mid C)$$
    - **Prior Probability $P(C)$**: The baseline probability of a message being Spam vs Ham.
    - **Likelihood $P(x_i \\mid C)$**: The probability of observing token $x_i$ given class $C$.
    - Laplace smoothing ($\alpha = 0.5$) is applied to handle previously unseen tokens.
    """)

    st.markdown("---")
    st.markdown("### 4️⃣ Frequently Asked Questions")
    faq1 = st.expander("❓ Why use TF-IDF instead of simple word counts?")
    faq1.write("Simple Bag-of-Words counts give undue advantage to common words. TF-IDF penalizes words that appear universally across all emails, highlighting words uniquely associated with spam or legitimate communication.")

    faq2 = st.expander("❓ Can this system run without internet access?")
    faq2.write("Yes! The entire NLP pipeline, tokenizer, vectorizer, and classifier operate 100% locally with zero external API calls or billing requirements.")
