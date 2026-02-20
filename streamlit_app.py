
import streamlit as st
from crayon import CrayonVocab

@st.cache_resource
def load_crayon_vocab_cached():
    vocab = CrayonVocab(device="auto")
    vocab.load_profile("lite")
    return vocab

vocab = load_crayon_vocab_cached()

st.set_page_config(page_title='CRAYON Tokenizer Demo', layout='wide')

st.title('CRAYON Tokenizer Demonstration')

default_text = "CRAYON is a hyper-fast tokenizer designed for modern AI. It offers unparalleled speed and efficiency in processing large volumes of text data."
user_text = st.text_area('Enter text here:', default_text, height=200)

if user_text:
    tokens_ids = vocab.tokenize(user_text)
    decoded_tokens = [vocab.decode([token_id]) for token_id in tokens_ids]

    word_count = len([word for word in user_text.split() if word.strip()])
    token_count = len(tokens_ids)

    colors = ["rgba(173, 216, 230, 0.4)", "rgba(144, 238, 144, 0.4)", "rgba(255, 255, 153, 0.4)", "rgba(255, 192, 203, 0.4)"]
    highlighted_tokens_html = []
    for i, token in enumerate(decoded_tokens):
        color = colors[i % len(colors)]
        highlighted_tokens_html.append(f"<span style='background-color: {color}; padding: 2px; margin: 0 1px;'>{token}</span>")

    display_tokens_html = "".join(highlighted_tokens_html)

    st.subheader('Tokenized Text')
    st.markdown(f"<div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px;'>{display_tokens_html}</div>", unsafe_allow_html=True)

    st.subheader('Word Count')
    st.write(word_count)

    st.subheader('Token Count')
    st.write(token_count)

    with st.expander("Show Detailed Token Information (IDs and Decoded Parts)"):
        st.write("--- ")
        st.markdown("### Token IDs:")
        st.code(str(tokens_ids))
        st.markdown("### Decoded Token Parts:")
        for i, token in enumerate(decoded_tokens):
            st.markdown(f"- ID: {tokens_ids[i]}, Part: `{token}`")
        st.write("--- ")

else:
    st.subheader('Tokenized Text')
    st.write("Please enter some text to tokenize.")
    st.subheader('Word Count')
    st.write("0")
    st.subheader('Token Count')
    st.write("0")
