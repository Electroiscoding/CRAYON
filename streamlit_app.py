import streamlit as st
from crayon import CrayonVocab
import time

st.set_page_config(page_title="CRAYON Tokenizer Demo", layout="wide")

st.title("🖍️ CRAYON Tokenizer Demo")
st.markdown("Interactive tokenization with CRAYON—the hyper-fast specialized tokenizer.")

# Initialize session state
if "vocab" not in st.session_state:
    with st.spinner("Loading vocabulary profile..."):
        st.session_state.vocab = CrayonVocab(device="cpu")  # Use CPU for cloud compatibility
        st.session_state.vocab.load_profile("lite")
    st.success("✓ Profile loaded!")

vocab = st.session_state.vocab

# User input
st.subheader("Input Text")
text_input = st.text_area(
    "Enter text to tokenize:",
    value="Hello, CRAYON! This is a production-grade tokenizer.",
    height=100
)

if text_input:
    # Tokenize
    start = time.perf_counter()
    tokens = vocab.tokenize(text_input)
    elapsed = (time.perf_counter() - start) * 1000
    
    # Decode
    decoded = vocab.decode(tokens)
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Tokens")
        st.code(str(tokens), language="python")
    
    with col2:
        st.subheader("Statistics")
        st.metric("Token Count", len(tokens))
        st.metric("Processing Time", f"{elapsed:.3f}ms")
    
    st.subheader("Decoded Output")
    st.write(decoded)
    
    # Token breakdown
    with st.expander("📋 Token Breakdown"):
        st.write(f"{'ID':<8} | {'Substring':<20}")
        st.write("-" * 30)
        for tid in tokens:
            substring = vocab.decode([tid])
            st.write(f"{tid:<8} | '{substring}'")
