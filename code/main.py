import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains import SequentialChain
from langchain_community.vectorstores import Chroma
from langchain.schema import SystemMessage
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPEN_API_KEY')

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables. Please set the OPENAI_API_KEY in your .env file.")

# Initialize ChatOpenAI model
model = ChatOpenAI(model="gpt-4", api_key=openai_api_key)
system_message = SystemMessage(content="you are a helpful AI assistant for sentiment analysis using customer feedback")

# Define a function to perform sentiment analysis
def analyze_sentiment(text):
    response = model.complete(prompt=f"Analyze the sentiment of this text: {text}")
    return response.choices[0].text.strip()

# Streamlit app
def main():
    st.title('Welcome to the Sentiment Analysis Tool')
    image_path = 'D:\\Ahmed\\LLM_sentimental_analysis\\smileys.jpg'
    try:
        image = Image.open(image_path)
        st.image(image, caption='Sentiment Analysis Visualization', use_column_width=True)
    except Exception as e:
        st.error(f"An error occurred when opening the image: {e}")

    user_input = st.text_area("Enter your feedback:", help="Type your feedback here and press enter.")
    if st.button("Analyze Feedback"):
        if user_input:
            sentiment_result = analyze_sentiment(user_input)
            st.success(f"Sentiment: {sentiment_result}")
        else:
            st.error("Please enter some feedback to analyze.")

    st.subheader("Bulk Sentiment Analysis from File")
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    file_path = os.path.join(parent_directory, 'Feedback_Dataset.xlsx')
    if st.button("Load Data"):
        data = pd.read_excel(file_path)
        st.dataframe(data)

if __name__ == "__main__":
    main()







