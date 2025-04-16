import streamlit as st
import torch
from model import SentimentTransformer

# --- Tokenizer mockup (replace with actual tokenizer if needed) ---
def mock_tokenizer(text, max_len=64):
    tokens = [ord(c) % 10000 for c in text]  # simple dummy tokenizer
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return tokens

# --- Load model ---
vocab_size = 10000
embed_dim = 128
num_heads = 4
hidden_dim = 256
num_classes = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentimentTransformer(vocab_size, embed_dim, num_heads, hidden_dim, num_classes).to(device)
model.load_state_dict(torch.load("sentiment_model.pth", map_location=device))
model.eval()

# --- Streamlit Page Config ---
st.set_page_config(page_title="Sentiment Classifier", page_icon="💬", layout="centered")

# --- Custom CSS ---
st.markdown("""
    <style>
    textarea {
        font-size: 16px !important;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5em 1em;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .result-box {
        border-radius: 10px;
        padding: 1em;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
    }
    .positive {
        background-color: #d4edda;
        color: #155724;
        animation: fadeIn 1s ease-in;
    }
    .negative {
        background-color: #f8d7da;
        color: #721c24;
        animation: fadeIn 1s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

# --- App Title ---
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>💬 Sentiment Classifier (Transformer)</h1>", unsafe_allow_html=True)

# --- User Input ---
user_input = st.text_area("📝 Enter your sentence to analyze:", height=150)

# --- Prediction Button ---
if st.button("🔍 Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a sentence.")
    else:
        tokens = mock_tokenizer(user_input)
        input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            sentiment = "Positive" if pred == 1 else "Negative"

            # --- Display result with color and emoji ---
            if sentiment == "Positive":
                st.markdown(f"<div class='result-box positive'>✅ Predicted Sentiment: <strong>{sentiment}</strong> 😊</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='result-box negative'>❌ Predicted Sentiment: <strong>{sentiment}</strong> 😞</div>", unsafe_allow_html=True)

# --- Info/Help section ---
st.markdown("---")
with st.expander("ℹ️ About this app"):
    st.markdown("""
    This is a simple yet powerful Transformer-based sentiment classifier built using PyTorch and Streamlit.

    **Features:**
    - Transformer model for deep context understanding
    - Color-coded and emoji-enhanced UI
    - Dummy tokenizer (can be replaced with BERT or custom)
    - Real-time text analysis
    
    _Try entering sentences like:_
    - "I love how this works!"
    - "This is the worst day ever."
    """)
