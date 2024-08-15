import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForTokenClassification

load_dotenv()
huggingface_token = os.getenv("HUGGINGFACE_API_TOKEN")
if huggingface_token:
    login(huggingface_token)
else:
    print("HUGGINGFACE_API_TOKEN not found in environment variables")

import streamlit as st
@st.cache_resource
# def load_model():
#     # model_path = "binary_model_xml-r_complete_dataset"
#     # tokenizer = AutoTokenizer.from_pretrained(model_path)
#     # model = AutoModelForTokenClassification.from_pretrained(model_path)
#     # return tokenizer, model

#     model_names = {
#         "XML-ROBERTa_Binary": "Rudra03/XML-ROBERTa_Binary",
#         "XML-ROBERTa_BIO": "Rudra03/XML-ROBERTa_BIO",
#         "M-Bert_Binary": "Rudra03/binary_m-bert",
#         "M-Bert_BIO": "Rudra03/bio_m-bert",
#         "Muril_Binary": "Rudra03/binary_muril",
#         "Muril_BIO": "Rudra03/bio_muril",
#         "Indic-Bert_Binary": "Rudra03/binary_indic-bert",
#         "Indic-Bert_BIO": "Rudra03/bio_indic-bert",
#     }
#     models = {}
#     tokenizers = {}
#     for name, path in model_names.items():
#         tokenizers[name] = AutoTokenizer.from_pretrained(path)
#         models[name] = AutoModelForTokenClassification.from_pretrained(path)
#     return tokenizers, models

def load_model(selected_model):
    model_names = {
        "XML-ROBERTa_Binary": "Rudra03/XML-ROBERTa_Binary",
        "XML-ROBERTa_BIO": "Rudra03/XML-ROBERTa_BIO",
        "M-Bert_Binary": "Rudra03/binary_m-bert",
        "M-Bert_BIO": "Rudra03/bio_m-bert",
        "Muril_Binary": "Rudra03/binary_muril",
        "Muril_BIO": "Rudra03/bio_muril",
        "Indic-Bert_Binary": "Rudra03/binary_indic-bert",
        "Indic-Bert_BIO": "Rudra03/bio_indic-bert",
    }
    tokenizer = AutoTokenizer.from_pretrained(model_names[selected_model])
    model = AutoModelForTokenClassification.from_pretrained(model_names[selected_model])
    return tokenizer, model

def predict_claims(text, _tokenizer, _model):
    # Tokenize the input text
    text = text.lower()
    text_tokens = text.split(" ")
    inputs = _tokenizer(text_tokens, return_tensors="pt",
                        truncation=True, padding=True, is_split_into_words=True, max_length=512)

    # Get model predictions
    with torch.no_grad():
        outputs = _model(**inputs)

    # Process the outputs
    predictions = torch.argmax(outputs.logits, dim=2)
    word_ids = inputs.word_ids()

    return word_ids, predictions[0].tolist()


def sub_token_to_word(word_ids, predictions):
    actual_predictions = [0]*(len(word_ids) + 1)
    for word_id, prediction in zip(word_ids, predictions):
        if word_id is None:
            continue
        elif prediction == 1:
            actual_predictions[word_id] = 1
        elif prediction == 2:
            actual_predictions[word_id] = 2
    return actual_predictions


def post_processing_Binary(predictions):
    count_zero = predictions.count(0)
    count_one = predictions.count(1)
    if count_zero <= 2:
        predictions = [1]*len(predictions)
    if count_one <= 2:
        predictions = [0]*len(predictions)

    return predictions


def post_processing_BIO(predictions):
    count_zero = predictions.count(0)
    count_two = predictions.count(2)
    # if count_zero <= 2:
    #     predictions = []*len(predictions)
    if count_two <= 2:
        predictions = [0]*len(predictions)

    return predictions


def highlight_claims(tokens, predictions):
    highlighted_text = []
    for token, pred in zip(tokens, predictions):
        if pred == 1:
            highlighted_text.append(
                f'<span style="background-color: #90EE70 ; font-weight: bold ; font-style: italic ; color: black; border-radius: 5px ; padding: 3px">{token}</span>')
        elif pred == 2:
            highlighted_text.append(
                f'<span style="background-color: #90EE70 ; font-weight: bold ; font-style: italic ; color: black; border-radius: 5px ; padding: 3px">{token}</span>')
        else:
            highlighted_text.append(token)

    return " ".join(highlighted_text)



def main():
    st.title("Claim Span Identification Tool")

    model_name = ["XML-ROBERTa", "M-Bert", "Muril", "Indic-Bert"]
    selected_model_name = st.selectbox("Select a model:", model_name)

    # Tagging type selection
    tagging_type = st.radio("Select tagging type:", ["Binary", "BIO"])

    selected_model = f"{selected_model_name}_{tagging_type}"

    # Text input
    text = st.text_area("Enter your text here:", height=150)

    if st.button("Detect Claims"):
        if text:
            with st.spinner("Detecting claims..."):
                #load the model
                tokenizer, model = load_model(selected_model)
                # tokenizer = tokenizers[selected_model]
                # model = models[selected_model]
                # tokenizer = AutoTokenizer.from_pretrained("Rudra03/binary_xlm-r")
                # model = AutoModelForTokenClassification.from_pretrained("Rudra03/binary_xlm-r")
                word_ids, predictions = predict_claims(text, tokenizer, model)
                actual_predictions = sub_token_to_word(word_ids, predictions)
                if tagging_type == "Binary":
                    post_processed_predictions = post_processing_Binary(
                        actual_predictions)
                else:
                    post_processed_predictions = post_processing_BIO(
                        actual_predictions)
                tokens = text.split(" ")
                highlighted_text = highlight_claims(
                    tokens, post_processed_predictions)
            # st.write(predictions)
            st.subheader("Text with Highlighted Claims:")
            st.markdown(highlighted_text, unsafe_allow_html=True)
        else:
            st.warning("Please enter some text to analyze.")


if __name__ == "__main__":
    main()
