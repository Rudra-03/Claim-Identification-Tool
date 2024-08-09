# Claim Span Identification Tool

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Models](#models)
7. [Technical Details](#technical-details)
8. [Contributing](#contributing)
9. [Acknowledgements](#acknowledgements)
10. [Contact](#contact)

## Overview

The Claim Span Identification Tool is a state-of-the-art natural language processing application designed to identify and highlight claim spans within text. Utilizing advanced transformer models, this tool offers multi-lingual support and flexible tagging schemes, making it suitable for a wide range of text analysis tasks.


## Features

- **Multiple Pre-trained Models**:
  - XML-ROBERTa
  - Multilingual BERT (M-BERT)
  - Multilingual Representations for Indian Languages (MuRIL)
  - IndicBERT
- **Dual Tagging Schemes**:
  - Binary (Claim/Non-Claim)
  - BIO (Begin-Inside-Outside)
- **Real-time Claim Span Highlighting**
- **Intuitive Web Interface**
- **Multi-lingual Support**
- **Optimized Performance**

## Requirements

- Python 3.7+
- PyTorch 1.7+
- Transformers 4.0+
- Streamlit 0.80+

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/[username]/claim-span-identification.git
   cd claim-span-identification


2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use \`venv\Scripts\activate\`

3. Install the required dependencies:
   ```
   pip install -r requirements.txt

## Usage

1. Launch the Streamlit application:
   ```
   streamlit run app.py

2. Load the App: Open the app in your browser at the provided URL after running the above command.

3. Model Selection: Use the dropdown menu to select the desired model architecture.

4. Tagging Type: Choose between Binary and BIO tagging methods.

5. Input Text: Enter or paste the text you want to analyze into the text area.

6. Detect Claims: Click the "Detect Claims" button to highlight claims in the text.

## Code Structure

- load_model(): Loads and caches the pre-trained models and tokenizers.
- predict_claims(): Tokenizes the input text and generates predictions using the selected model.
- sub_token_to_word(): Converts sub-token level predictions to word level.
- post_processing_Binary(): Post-processes binary predictions.
- post_processing_BIO(): Post-processes BIO predictions.
- highlight_claims(): Highlights the detected claims in the text.
- main(): The main function that sets up the Streamlit interface and handles user inputs.

## Models

Our tool incorporates the following state-of-the-art transformer models:

| Model | Type | Languages Supported |
|-------|------|---------------------|
| XML-ROBERTa | Binary, BIO | 100+ languages |
| M-BERT | Binary, BIO | 104 languages |
| MuRIL | Binary, BIO | 17 Indian languages + English |
| IndicBERT | Binary, BIO | 12 major Indian languages |


## Technical Details

- **Framework**: PyTorch
- **Tokenization & Modeling**: Hugging Face Transformers
- **Web Interface**: Streamlit
- **Caching**: Streamlit's `@st.cache_resource` for optimized model loading
- **Post-processing**: Custom algorithms for refining model predictions


## Contributing

I welcome contributions to enhance the functionality and performance of this tool. Please follow these steps:

1. Fork the repository
2. Create your feature branch (\`git checkout -b feature/AmazingFeature\`)
3. Commit your changes (\`git commit -m 'Add some AmazingFeature'\`)
4. Push to the branch (\`git push origin feature/AmazingFeature\`)
5. Open a Pull Request

## Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Streamlit](https://www.streamlit.io/)
- [PyTorch](https://pytorch.org/)

## Contact

Rudra Roy - [royrudra164@gmail.com](mailto:royrudra164@gmail.com)

<!-- Project Link: [https://github.com/[username]/claim-span-identification](https://github.com/[username]/claim-span-identification) -->

