import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains import SequentialChain
from langchain_community.vectorstores import Chroma
from langchain.schema import SystemMessage, HumanMessage
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPEN_API_KEY')

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables. Please set the OPENAI_API_KEY in your .env file.")

persistent_directory = os.path.join(os.getcwd(), "chroma_db")
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=openai_api_key
)

try:
    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embeddings
    )
    # If successful, you can now retrieve from db
    print("Chroma DB loaded successfully!")
except Exception as e:
    print(f"Could not load Chroma DB: {e}")
    db = None

# Prompt-engineered system message
prompt_engineered_system = """You are a helpful assistant specialized in sentiment analysis for e-commerce feedback.
For any given feedback, respond with:
1) Overall Sentiment: Positive, Negative, or Neutral
2) A short explanation (1-2 sentences)
3) A numeric sentiment score from 1-5 (1 = very negative, 3 = neutral, 5 = very positive)

Format your response exactly like this (no extra text):
Sentiment: <Positive/Negative/Neutral>
Explanation: <short explanation>
Score: <1-5>
"""

system_message_pe = SystemMessage(content=prompt_engineered_system)

model_pe = ChatOpenAI(
    model="gpt-4o",
    api_key=openai_api_key,
    temperature=0.0
)
def retrieve_product_context(user_feedback: str, top_k=2) -> str:
    """
    Query the Chroma vector store for the top-k relevant chunks
    that might help GPT better understand or reference the products.
    """
    if not db:
        # If db failed to load, just return empty context
        return ""

    docs = db.similarity_search(user_feedback, k=top_k)
    # Combine the text of these docs
    context = "\n".join([doc.page_content for doc in docs])
    return context


#def analyze_sentiment_prompt_engineered(text: str) -> str:
#   """Send the feedback text to GPT with the prompt-engineered system message."""
#    user_message = HumanMessage(content=f"Feedback: {text}")
#    response = model_pe([system_message_pe, user_message])
#    return response.content.strip()

def analyze_sentiment_prompt_engineered(text: str) -> str:
    """Send the feedback text to GPT with the prompt-engineered system message."""
    prompt = (
            f"""Given the customer feedback below, identify:
    1. Sentiment (positive, neutral, or negative)
    2. Explanation for your sentiment
    3. A numeric score (1-5)
    4. Feedback category (one of: Pricing, Shipment delays, Delivery issues, service, Technical issues, Other)
    
    Feedback: {text}
    Answer:"""
        )

    user_message = HumanMessage(content=prompt)
    response = model_pe([system_message_pe, user_message])
    return response.content.strip()

def extract_sentiment_fields(response: str):
    # This is a quick parsing approach — adjust if the formatting changes
    try:
        lines = response.split("\n")
        sentiment = next((line.split(":")[1].strip() for line in lines if "Sentiment" in line), "Unknown")
        explanation = next((line.split(":")[1].strip() for line in lines if "Explanation" in line), "")
        score = next((line.split(":")[1].strip() for line in lines if "Score" in line), "0")
        category = next((line.split(":")[1].strip() for line in lines if "Category" in line), "Other")
        return sentiment, explanation, score, category
    except Exception as e:
        print("Parsing error:", e)
        return "Unknown", "", "0", "Other"

def generate_wordcloud(text_data: str):
    """Generate a word cloud from text_data."""
    wc = WordCloud(width=800, height=400, background_color="white").generate(text_data)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    return fig

def display_star_rating(score: float, max_stars=5) -> str:
    """
    Convert a numeric score (1–5) into a string of star icons.
    Example: 4 -> "★★★★☆"
    """
    rating = int(round(score))
    rating = min(max(rating, 1), max_stars)
    filled_star = "★"
    empty_star = "☆"
    return filled_star * rating + empty_star * (max_stars - rating)

def display_smiley_rating(score: float) -> str:
    rating = int(round(score))
    rating = max(1, min(rating, 5))
    rating_to_emoji = {
        1: "😡",
        2: "😟",
        3: "😐",
        4: "😊",
        5: "😍"
    }
    return rating_to_emoji.get(rating, "🤔")

PRODUCTS = [
    "BrewoMatic Steam Pro 3000",
    "SkyTune Sonic Pro X1",
    "DustMaster Ultra 500",
    "QuantumTech A12 Notebook",
    "HyperKlick FX-21"
]

# We can let the user pick a product from the sidebar
st.sidebar.title("Product Info")
selected_product = st.sidebar.selectbox("Select a product", PRODUCTS)

# Provide a text input for user to type a question or query about that product
user_query = st.sidebar.text_input("Ask about this product:")
if st.sidebar.button("Search Product Info"):
    if user_query.strip():
        retrieved_docs = retrieve_product_context(user_query)
        st.sidebar.write("### Relevant Chunks:")
        for i, doc in enumerate(retrieved_docs, start=1):
            st.sidebar.write(f"**Document {i}:**\n{doc.page_content}")
            if doc.metadata:
                st.sidebar.write(f"_Source_: {doc.metadata.get('source', 'Unknown')}")
            st.sidebar.write("---")
    else:
        st.sidebar.warning("Please enter a query to search for product info.")

def main():
    st.title('Prompt-Engineered GPT: Sentiment Analysis')

    image = 'smileys.jpg'
    image_path = os.path.join(os.getcwd(), '..//', image)
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption='Sentiment Analysis', use_column_width=True)

    st.markdown("""
    **How this works:**
    - **Single Feedback Mode**: Enter one feedback:
      - A sentiment label (Positive/Negative/Neutral)
      - A short explanation
      - A numeric score (1–5)
      - Star rating & a smiley
    - **Bulk Analysis Mode**: Upload a file of multiple feedbacks:
      - Analyze each row with GPT
      - Histogram of numeric scores
      - A pie chart of sentiment labels
      - A word cloud of the text
    """)

    # --- SINGLE FEEDBACK ---
    st.subheader("Single Feedback Analysis")

    user_input = st.text_area("Enter a single feedback here:")
    if st.button("Analyze Feedback"):
        if user_input.strip():
            # Call GPT
            gpt_output = analyze_sentiment_prompt_engineered(user_input)
            # Parse it
            sentiment, explanation, score_str, category = extract_sentiment_fields(gpt_output)

            st.write("**GPT Output**:")
            st.write(gpt_output)
            st.write(f"**Category**: {category}")

            # Show structured results if they exist
            if sentiment != "Unknown":
                st.write(f"**Sentiment**: {sentiment}")
                st.write(f"**Explanation**: {explanation}")
                st.write(f"**Numeric Score**: {score_str}")

                # Display star rating and smiley
                try:
                    score_val = float(score_str)
                    star_str = display_star_rating(score_val)
                    smiley_str = display_smiley_rating(score_val)

                    st.write(f"**Star Rating**: {star_str}")
                    st.write(f"**Smiley**: {smiley_str}")
                except ValueError:
                    st.error("Could not convert GPT's 'Score' to a numeric value.")
        else:
            st.warning("Please enter some text before clicking 'Analyze Feedback'.")

    # --- BULK ANALYSIS ---
    st.subheader("Bulk Analysis from Excel/CSV")
    uploaded_file = st.file_uploader("Upload your file (.xlsx or .csv)", type=["xlsx", "csv"])
    if uploaded_file is not None:
        # Read into DataFrame
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        st.write("**Sample of your data** (first 5 rows):")
        st.dataframe(df.head())

        # Select the text column for analysis
        text_col = st.selectbox("Select the column containing feedback text:", df.columns)

        if st.button("Run Bulk GPT Analysis"):
            data_copy = df.copy()

            sentiments = []
            explanations = []
            scores = []
            categories = []

            for _, row in data_copy.iterrows():
                feedback_text = str(row[text_col])
                result = analyze_sentiment_prompt_engineered(feedback_text)
                s, e, sc, cat = extract_sentiment_fields(result)
                sentiments.append(s)
                explanations.append(e)
                scores.append(sc)
                categories.append(cat)

            data_copy["GPT_Sentiment"] = sentiments
            data_copy["GPT_Explanation"] = explanations
            data_copy["GPT_Score"] = scores
            data_copy["GPT_Category"] = categories

            st.write("**Analysis Results (all rows)**:")
            st.dataframe(data_copy)

            # Convert GPT_Score to numeric
            data_copy["GPT_Score_Num"] = pd.to_numeric(data_copy["GPT_Score"], errors='coerce')

            # 1) Histogram of numeric scores
            st.subheader("Histogram of GPT-Assigned Scores")
            fig_hist, ax_hist = plt.subplots()
            data_copy["GPT_Score_Num"].plot(kind='hist', bins=5, rwidth=0.8, ax=ax_hist)
            ax_hist.set_xlabel("Sentiment Score (1=Neg, 5=Pos)")
            ax_hist.set_ylabel("Count")
            ax_hist.set_title("Distribution of GPT-Assigned Scores")
            st.pyplot(fig_hist)

            st.subheader("Pie Chart of Sentiment Labels")
            label_counts = data_copy["GPT_Sentiment"].value_counts()
            fig_pie, ax_pie = plt.subplots()
            ax_pie.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140)
            ax_pie.axis("equal")
            st.pyplot(fig_pie)

            st.subheader("Category Breakdown")
            category_counts = data_copy["GPT_Category"].value_counts()
            fig_cat, ax_cat = plt.subplots()
            ax_cat.bar(category_counts.index, category_counts.values, color='skyblue')
            ax_cat.set_title("Feedback Categories")
            ax_cat.set_xlabel("Category")
            ax_cat.set_ylabel("Count")
            st.pyplot(fig_cat)

            st.subheader("Word Cloud of Feedback")
            all_text = " ".join(data_copy[text_col].astype(str).tolist())
            wc_fig = generate_wordcloud(all_text)
            st.pyplot(wc_fig)

            st.success("Bulk analysis completed!")

if __name__ == "__main__":
    main()







