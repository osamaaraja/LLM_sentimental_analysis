import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from langchain_community.vectorstores import Chroma
from langchain.schema import SystemMessage, HumanMessage
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import re
nltk.download('stopwords')


load_dotenv()
openai_api_key = os.getenv('OPEN_API_KEY')

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables. Please set the OPENAI_API_KEY in your .env file.")

persistent_directory = os.path.join(os.getcwd(),"..//", "chroma_db")
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=openai_api_key
)

try:
    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embeddings
    )

    print("Chroma DB loaded successfully!")
except Exception as e:
    print(f"Could not load Chroma DB: {e}")
    db = None

prompt_engineered_system = """You are a helpful assistant specialized in sentiment analysis for e-commerce feedback.
For any given feedback, respond with:
1) Sentiment: Positive, Neutral, or Negative
2) Explanation: (1-2 sentences)
3) A numeric score: (1-5)
4) Feedback category: (one of: Pricing, Shipment delays, Delivery issues, Service, Technical issues, Product assortment, Packaging, Employees behavior, Product quality, Other)

Format your response exactly like this (no extra text):
Sentiment: <Positive/Negative/Neutral>
Explanation: <short explanation>
Score: <1-5>
Category: <Pricing/Shipment delays/Delivery issues/Service/Technical issues/Product assortment/Packaging/Employees behavior/Product quality/Other>
"""
system_message_pe = SystemMessage(content=prompt_engineered_system)

model_pe = ChatOpenAI(
    model="gpt-4o",
    api_key=openai_api_key,
    temperature=0.0
)

def retrieve_product_docs(user_query: str, product_name: str, top_k=5) -> list:
    if not db:
        return []
    # Build a retriever that uses the filter
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": top_k,
            "filter": {"product": product_name}  # EXACT match
        }
    )
    docs = retriever.get_relevant_documents(user_query)
    return docs

def analyze_sentiment_prompt_engineered(text: str) -> str:
    user_message = HumanMessage(content=f"Feedback: {text}")
    response = model_pe([system_message_pe, user_message])
    return response.content.strip()

def extract_sentiment_fields(gpt_output: str):

    sentiment = "Unknown"
    explanation = ""
    score = ""
    category = "Other"

    lines = gpt_output.splitlines()
    for line in lines:
        lower_line = line.lower().strip()
        if lower_line.startswith("sentiment:"):
            sentiment = line.split(":", 1)[1].strip()
        elif lower_line.startswith("explanation:"):
            explanation = line.split(":", 1)[1].strip()
        elif lower_line.startswith("score:"):
            score = line.split(":", 1)[1].strip()
        elif lower_line.startswith("category:"):
            category = line.split(":", 1)[1].strip()

    return sentiment, explanation, score, category

def clean_text(text):

    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    stopwords_list = set(stopwords.words("english"))
    tokens = [tok for tok in tokens if tok not in stopwords_list]
    tokens = [tok for tok in tokens if len(tok) > 2]

    return tokens
def generate_cleaned_wordcloud(feedbacks):

    all_tokens = []
    for line in feedbacks:
        tokens = clean_text(line)
        all_tokens.extend(tokens)

    cleaned_text = " ".join(all_tokens)

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        collocations=False,
        min_font_size=10
    ).generate(cleaned_text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    return fig

def display_logo_and_header(logo_path: str, logo_width: int = 80):

    if not os.path.exists(logo_path):
        st.warning(f"Logo file not found: {logo_path}")
        return
    col1, col2 = st.columns([1, 4])

    with col1:
        st.image(logo_path, width=logo_width)

    with col2:
        st.write("")

def display_star_rating(score: float, max_stars=5) -> str:

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
    "HyperKlick FX-21",
    "NanoMouse Prime XR"
]

st.sidebar.title("Product Info")
selected_product = st.sidebar.selectbox("Select a product", PRODUCTS)

# Load product docs when button is clicked
if st.sidebar.button("Search Product Info"):
    docs = retrieve_product_docs("Product specs", selected_product, top_k=5)

    if docs:
        st.sidebar.write("### Relevant Chunks:")
        for i, doc in enumerate(docs, start=1):
            st.sidebar.write(f"**Document {i}:**\n{doc.page_content}")
            if doc.metadata:
                st.sidebar.write(f"_Source_: {doc.metadata.get('source', 'Unknown')}")
            st.sidebar.write("---")
    else:
        st.sidebar.warning("No relevant documents found.")

# Input for user query and a new button for submission
user_query = st.sidebar.text_input("Ask about this product:")

if st.sidebar.button("Submit Query"):
    if user_query.strip():
        docs = retrieve_product_docs(user_query, selected_product, top_k=5)

        if docs:
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = f"""Answer the following question using the product context below:\n\nContext:\n{context}\n\nQuestion: {user_query}"""
            response = model_pe([SystemMessage(content="You are a helpful assistant."), HumanMessage(content=prompt)])
            st.sidebar.markdown("### Answer:")
            st.sidebar.markdown(response.content.strip())
        else:
            st.sidebar.warning("No relevant documents found.")
    else:
        st.sidebar.warning("Please enter a query to submit.")

def main():
    st.title('Prompt-Engineered GPT: Sentiment Analysis')

    image = 'OTH_Logo_farbig__zweizeilig_.jpg'
    image_path = os.path.join(os.getcwd(), '..//', image)

    display_logo_and_header(image_path, logo_width=150)

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

            gpt_output = analyze_sentiment_prompt_engineered(user_input)
            sentiment, explanation, score_str, category = extract_sentiment_fields(gpt_output)

            st.write("**GPT Output**:")
            st.write(gpt_output)
            #st.write(f"**Category**: {category}") # additional printing of the category

            if sentiment != "Unknown":
                st.write(f"**Sentiment**: {sentiment}")
                st.write(f"**Explanation**: {explanation}")
                st.write(f"**Numeric Score**: {score_str}")

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
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        # Select how many rows of data need to be evaluated
        df = df.sample(n=500, random_state=42).reset_index(drop=True)

        st.write("**Sample of your data** (first 5 rows):")
        st.dataframe(df.head())

        text_col = st.selectbox("Select the column containing feedback text:", df.columns)

        if st.button("Run Bulk GPT Analysis"):
            data_copy = df.copy()

            sentiments = []
            explanations = []
            scores = []
            categories = []

            progress_bar = st.progress(0)
            status_text = st.empty()

            total_rows = len(data_copy)

            for idx, row in data_copy.iterrows():
                feedback_text = str(row[text_col])
                result = analyze_sentiment_prompt_engineered(feedback_text)
                s, e, sc, cat = extract_sentiment_fields(result)

                sentiments.append(s)
                explanations.append(e)
                scores.append(sc)
                categories.append(cat)

                # Update progress bar and status
                progress = (idx + 1) / total_rows
                progress_bar.progress(progress)
                status_text.text(f"Processing {idx + 1}/{total_rows} entries...")

            data_copy["GPT_Sentiment"] = sentiments
            data_copy["GPT_Explanation"] = explanations
            data_copy["GPT_Score"] = scores
            data_copy["GPT_Category"] = categories

            st.write("**Analysis Results (all rows)**:")
            st.dataframe(data_copy)

            data_copy["GPT_Score_Num"] = pd.to_numeric(data_copy["GPT_Score"], errors='coerce')

            st.subheader("Histogram of GPT-Assigned Scores")
            fig_hist, ax_hist = plt.subplots()
            data_copy["GPT_Score_Num"].plot(kind='hist', bins=5, rwidth=0.8, ax=ax_hist)
            ax_hist.set_xlabel("Sentiment Score (1=Neg, 5=Pos)")
            ax_hist.set_ylabel("Count")
            ax_hist.set_title("Distribution of GPT-Assigned Scores")
            st.pyplot(fig_hist)
            fig_hist.savefig("histogram.pdf", format="pdf")

            st.subheader("Pie Chart of Sentiment Labels")
            label_counts = data_copy["GPT_Sentiment"].value_counts()
            fig_pie, ax_pie = plt.subplots()
            ax_pie.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140)
            ax_pie.axis("equal")
            st.pyplot(fig_pie)
            fig_pie.savefig("pie_chart.pdf", format="pdf")

            st.subheader("Category Breakdown")
            category_counts = data_copy["GPT_Category"].value_counts()
            fig_cat, ax_cat = plt.subplots()
            ax_cat.bar(category_counts.index, category_counts.values, color='skyblue')
            ax_cat.set_title("Feedback Categories")
            ax_cat.set_xlabel("Category")
            ax_cat.set_ylabel("Count")
            ax_cat.tick_params(axis='x', labelrotation=45)  # Rotate x-axis labels
            fig_cat.tight_layout()
            st.pyplot(fig_cat)
            fig_cat.savefig("category_breakdown.pdf", format="pdf")

            st.subheader("Word Cloud")

            feedback_lines = []
            for idx, row in data_copy.iterrows():
                original_text = str(row[text_col])
                category = str(row["GPT_Category"])
                category = category.replace(" ", "_").lower()

                combined_text = original_text + " " + category
                feedback_lines.append(combined_text)

            fig_wc = generate_cleaned_wordcloud(feedback_lines)
            st.pyplot(fig_wc)
            fig_wc.savefig("word_cloud.pdf", format="pdf")

            st.success("Bulk analysis completed!")

if __name__ == "__main__":
    main()







