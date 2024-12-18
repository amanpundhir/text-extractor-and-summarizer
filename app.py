import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# TextRank-based summarization function
def summarize_text(text, num_sentences=3):
    if not text.strip():
        return "No text to summarize!"
    
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)

# Title of the app
st.title("Image to Text Extractor and Summarizer")

# File uploader to upload an image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Check if a file is uploaded
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Convert the uploaded file to a PIL Image
    image = Image.open(uploaded_file)
    image_np = np.array(image)  # Convert to numpy array (for OpenCV)

    # Convert image to grayscale for better OCR accuracy
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Perform OCR using pytesseract
    extracted_text = pytesseract.image_to_string(gray_image)

    # Display the extracted text
    st.subheader("Extracted Text:")
    st.text(extracted_text)

    # Add a button to summarize the text
    if st.button("Summarize Text"):
        if extracted_text.strip():  # Ensure there's text to summarize
            st.subheader("Summary:")
            summary = summarize_text(extracted_text, num_sentences=3)
            st.write(summary)
        else:
            st.warning("No text extracted to summarize!")
else:
    st.info("Please upload an image to extract text.")
