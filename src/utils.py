"""
Utility functions and sample presets for AI Email Spam Detector.
"""

import re
from typing import Dict, Any, List

SAMPLE_MESSAGES = {
    "🎉 Lottery & Prize Scam (Spam)": (
        "CONGRATULATIONS! You have won a $1,000 Walmart Gift Card! Click here to claim your prize "
        "now before midnight: http://gift-reward-winner.xyz/claim. Reply with your bank details to complete transfer."
    ),
    "🔒 Urgent Account Security Alert (Spam)": (
        "SECURITY ALERT: Your PayPal account has been temporarily suspended due to unauthorized login attempts. "
        "Verify your credentials within 24 hours at http://paypal-security-update.com/login or your account will be deleted."
    ),
    "📈 Crypto Investment Scam (Spam)": (
        "Guaranteed 500% returns in 24 hours! Invest $100 in Bitcoin and receive $500 daily passive income. "
        "Join our VIP Telegram channel immediately: t.me/crypto_fast_profit."
    ),
    "👥 Work Sprint Review (Not Spam)": (
        "Hi team, please find attached the meeting minutes and agenda for our weekly sprint review tomorrow "
        "at 10:00 AM in Conference Room B. Let me know if you have any updates to the backlog."
    ),
    "🎓 University Assignment Reminder (Not Spam)": (
        "Dear Students, please be reminded that Assignment 3 for Natural Language Processing is due this Sunday "
        "at 11:59 PM on Canvas. Office hours are available Wednesday from 2 PM to 4 PM."
    ),
    "✈️ Flight Confirmation (Not Spam)": (
        "Your flight booking to San Francisco (Booking Ref: W84KLP) is confirmed. Your e-ticket receipt is attached. "
        "Please arrive at the terminal 2 hours prior to departure. Safe travels with United Airlines."
    )
}

SPAM_TRIGGER_KEYWORDS = {
    'congratulations', 'won', 'winner', 'prize', 'lottery', 'claim', 'cash', 'free',
    'guaranteed', 'invest', 'bitcoin', 'crypto', 'urgent', 'suspended', 'immediately',
    'passcode', 'password', 'expire', 'limited', 'offer', 'gift', 'bonus', 'moneymarker',
    'urlmarker', 'phonemarker', 'commission', 'payout', 'giveaway', 'discount', 'refund'
}


def compute_text_stats(text: str) -> Dict[str, Any]:
    """
    Computes linguistic and structural properties of raw text.
    """
    if not isinstance(text, str):
        text = ""

    total_chars = len(text)
    total_chars_no_spaces = len(text.replace(" ", ""))
    words = text.split()
    total_words = len(words)
    
    # Sentences approx
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    sentence_count = max(len(sentences), 1 if total_words > 0 else 0)
    
    # Uppercase letters
    uppercase_chars = sum(1 for c in text if c.isupper())
    uppercase_percentage = (uppercase_chars / max(total_chars, 1)) * 100
    
    # Digits
    digits_count = sum(1 for c in text if c.isdigit())
    
    # Special symbols & Punctuation
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    
    # URLs and Emails
    urls_found = re.findall(r'https?://\S+|www\.\S+', text)
    emails_found = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    currency_found = re.findall(r'[\$£€¥₹]\s*\d+', text)
    
    # Readability estimate (Average word length)
    avg_word_length = (total_chars_no_spaces / total_words) if total_words > 0 else 0.0

    return {
        "total_chars": total_chars,
        "total_words": total_words,
        "sentence_count": sentence_count,
        "uppercase_percentage": round(uppercase_percentage, 1),
        "digits_count": digits_count,
        "special_chars": special_chars,
        "urls_count": len(urls_found),
        "emails_count": len(emails_found),
        "currency_count": len(currency_found),
        "avg_word_length": round(avg_word_length, 1),
        "urls_list": urls_found
    }


def generate_explanation(
    prediction: str,
    confidence: float,
    stats: Dict[str, Any],
    top_terms: List[str]
) -> List[str]:
    """
    Generates beginner-friendly bullet points explaining why the model made its prediction.
    """
    reasons = []

    if prediction == "SPAM":
        reasons.append(
            f"The model detected strong spam patterns with **{confidence:.1f}% confidence** using statistical feature weighting."
        )
        if top_terms:
            terms_str = ", ".join([f"`{term}`" for term in top_terms[:5]])
            reasons.append(f"Key suspicious/promotional keywords identified: {terms_str}.")
        
        if stats.get("urls_count", 0) > 0:
            reasons.append(f"Contains {stats['urls_count']} external link(s), a frequent indicator in phishing/scam emails.")
            
        if stats.get("currency_count", 0) > 0:
            reasons.append(f"Contains financial/currency terms or price promises ({stats['currency_count']} found).")
            
        if stats.get("uppercase_percentage", 0) > 15.0:
            reasons.append(f"High uppercase letter ratio ({stats['uppercase_percentage']}%), often used to create false urgency.")
            
        if stats.get("special_chars", 0) > 5:
            reasons.append(f"Contains noticeable exclamation marks or unusual punctuation symbols.")
            
    else:
        reasons.append(
            f"The message matches typical legitimate communication patterns with **{confidence:.1f}% confidence**."
        )
        if top_terms:
            terms_str = ", ".join([f"`{term}`" for term in top_terms[:5]])
            reasons.append(f"Key conversational/formal terms identified: {terms_str}.")
            
        if stats.get("urls_count", 0) == 0:
            reasons.append("No suspicious links or redirects were found in the text.")
            
        if stats.get("uppercase_percentage", 0) <= 15.0:
            reasons.append("Sentence casing is standard and does not show false-urgency capitalization.")
            
        reasons.append("Tone and vocabulary align with standard business, educational, or personal correspondence.")

    return reasons
