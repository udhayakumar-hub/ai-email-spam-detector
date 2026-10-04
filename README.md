# 🛡️ AI Email Spam Detector

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Natural Language Processing (NLP) web application that classifies incoming emails and SMS messages as **SPAM** or **NOT SPAM (HAM)** using TF-IDF vectorization and Machine Learning classifiers.

This application is **Application 3** of the NLP Project Series, built to be robust, self-contained, offline-ready, and beginner-friendly for college demonstrations and GitHub portfolios.

---

## 📌 Problem Statement

Unsolicited and malicious messages (spam, phishing scams, lottery frauds, suspicious link distributions) account for a significant volume of digital communication. They pose severe security risks to users, ranging from identity theft and credential compromise to malware installation. 

Traditional rule-based keyword filters fail against evolving spam vocabularies and subtle deceptive phrasing. This project implements a statistical and probabilistic NLP pipeline to automatically distinguish between legitimate communications and spam.

---

## 🎯 Objective

- Build a production-grade, local, and zero-cost spam classification web application.
- Implement an authentic text preprocessing and feature extraction pipeline using **TF-IDF (Term Frequency - Inverse Document Frequency)**.
- Train, evaluate, and compare **Multinomial Naive Bayes** and **Logistic Regression** classifiers.
- Provide explainable predictions with confidence percentages, linguistic statistics, and key trigger words.
- Deliver an interactive, modern Streamlit user interface suitable for academic demonstration.

---

## ✨ Key Features

- 🔍 **Real-Time Spam & Ham Detection**: Fast and accurate inference on raw email texts.
- ⚡ **1-Click Preset Samples**: Test realistic pre-configured spam (lottery, crypto, phishing) and ham (work sprint, college assignment, flight receipts) samples instantly.
- 🎯 **Confidence Scores & Explanations**: Visual confidence progress bars with clear explanations of why an email was classified as spam or ham.
- 📊 **Linguistic & Structural Insights**: Computes word count, character count, uppercase ratio, currency markers, and detected hyperlinks.
- 📈 **Model Evaluation Suite**: Interactive confusion matrix heatmap, Accuracy, Precision, Recall, and F1-score evaluation metrics on unseen test data.
- 🔄 **Algorithm Switching**: Seamlessly toggle between **Multinomial Naive Bayes** and **Logistic Regression**.
- 📋 **Session History & Analytics Dashboard**: Live KPI cards (Total Analyzed, Spam Count, Ham Count, Spam Rate) and a distribution pie chart.
- 📚 **Interactive NLP & ML Guide**: Beginner-friendly explanations of TF-IDF formulas, Bayes' Theorem, and text normalization.
- 🔒 **100% Offline & Private**: Zero API keys or cloud dependencies required; messages are processed strictly in local session memory.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **Streamlit** | Interactive modern web frontend & dashboard |
| **scikit-learn** | TF-IDF vectorization, Multinomial Naive Bayes, Logistic Regression & evaluation metrics |
| **NLTK** | Tokenization, English stopwords filtering, and Porter Stemmer |
| **Pandas & NumPy** | Data manipulation and numerical computations |
| **Matplotlib & Seaborn** | Confusion matrix heatmaps and session distribution charts |
| **pytest & unittest** | Unit testing and pipeline validation |

---

## 🔬 NLP & Machine Learning Workflow

```mermaid
flowchart LR
    A[Raw Text Input] --> B[Text Cleaning & Regex Normalization]
    B --> C[Stopword Removal & Porter Stemming]
    C --> D[TF-IDF Vectorizer (Unigrams & Bigrams)]
    D --> E[Multinomial Naive Bayes / Logistic Regression]
    E --> F[Spam / Ham Prediction + Confidence % + Trigger Analysis]
```

1. **Text Preprocessing**:
   - Case folding to lowercase.
   - Entity marker normalization for URLs (`urlmarker`), emails (`emailmarker`), phone numbers (`phonemarker`), and currency amounts (`moneymarker`).
   - Removal of non-alphanumeric punctuation.
   - Stopword filtering and word reduction using **Porter Stemmer**.

2. **Feature Extraction (TF-IDF)**:
   - Evaluates token frequencies across documents while downweighting universally frequent terms:
     $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \left(\log\frac{1 + N}{1 + \text{DF}(t)} + 1\right)$$

3. **Classification**:
   - **Multinomial Naive Bayes**:
     $$P(\text{Spam} \mid X) \propto P(\text{Spam}) \prod_{i=1}^{n} P(x_i \mid \text{Spam})$$
   - Optional **Logistic Regression** for linear decision boundaries.

---

## 📂 Project Structure

```
ai-email-spam-detector/
│
├── app.py                  # Main Streamlit web application & UI tabs
├── requirements.txt        # Python package dependencies
├── README.md               # Comprehensive project documentation
├── .gitignore              # Standard Python git ignore rules
│
├── data/
│   └── spam_dataset.csv    # Curated, realistic spam & ham dataset
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # NLP text cleaning, normalization, stemming & stopwords
│   ├── model.py            # SpamClassifier ML pipeline, training & evaluation
│   └── utils.py            # Text stats, sample messages & explanation generator
│
└── tests/
    ├── __init__.py
    └── test_model.py       # Unit tests for preprocessing, models & edge cases
```

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/ai-email-spam-detector.git
cd ai-email-spam-detector
```

### 2. Create and activate a virtual environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🖥️ How to Run the Application

Launch the Streamlit web application:
```bash
streamlit run app.py
```

The app will start and open automatically in your default browser at:
```
http://localhost:8501
```

---

## 🧪 Running Unit Tests

Run the test suite to verify preprocessing, model training, and predictions:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

Or using `pytest`:
```bash
pytest
```

---

## 💡 Example Usage

### Example 1: Phishing Email (Spam)
- **Input**:
  > *"SECURITY ALERT: Your PayPal account has been temporarily suspended due to unauthorized login attempts. Verify identity: http://paypal-security-update.com/login"*
- **Result**: `🚨 SPAM (Confidence: 98.4%)`
- **Key Indicators**: Contains external link `urlmarker`, words `["security", "alert", "suspend", "verifi"]`.

### Example 2: College Notification (Not Spam)
- **Input**:
  > *"Dear Students, please be reminded that Assignment 3 for Natural Language Processing is due this Sunday at 11:59 PM on Canvas."*
- **Result**: `✅ NOT SPAM (Confidence: 96.2%)`
- **Key Indicators**: Clean sentence casing, educational context, no urgent financial or reward promises.

---

## 🔮 Future Improvements

- 🤖 Add Transformer-based embeddings (e.g., DistilBERT / RoBERTa) for contextual zero-shot classification.
- 📎 Implement `.eml` and `.msg` file upload parsing to classify entire raw email files.
- 🌐 Add multi-lingual spam detection for regional language messages.
- 📧 Integration with local email clients via IMAP/POP3 for live mailbox filtering.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
