import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv('OPEN_API_KEY')
if not openai_api_key:
    raise ValueError("OpenAI API key not found.")

folder_name = "company_products"
current_dir = os.getcwd()
folder_path = os.path.join(current_dir, "..", folder_name)

# 2) The path to your Chroma DB
persistent_directory = os.path.join(current_dir, "..", "chroma_db")

product_map = {
    "brewmatic": "BrewoMatic Steam Pro 3000",
    "skytune": "SkyTune Sonic Pro X1",
    "dustmaster": "DustMaster Ultra 500",
    "quantumtech": "QuantumTech A12 Notebook",
    "hyperklick": "HyperKlick FX-21",
    "nano_mouse": "NanoMouse Prime XR"
}

file_paths = []
for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):
        file_paths.append(os.path.join(folder_path, file_name))

if not file_paths:
    raise ValueError(f"No .txt files found in '{folder_path}'.")

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
all_docs = []

for path in file_paths:
    print(f"Loading file: {path}")
    loader = TextLoader(path, encoding="utf-8")
    documents = loader.load()
    base_name = os.path.basename(path).lower()
    matched_product = "UnknownProduct"
    for key, val in product_map.items():
        if key in base_name:
            matched_product = val
            break

    updated_docs = []
    for doc in documents:
        doc.metadata["product"] = matched_product
        updated_docs.append(doc)

    split_docs = text_splitter.split_documents(updated_docs)
    all_docs.extend(split_docs)

    print("\n--- Debug: Listing metadata and partial content of each chunk ---")
    for doc in split_docs:
        print(f"Product: {doc.metadata.get('product', 'No product found')} | Content snippet: {doc.page_content[:200]}")

print(f"\nTotal chunks across all files: {len(all_docs)}")

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api_key)
print("\nCreating Chroma vector store...")

db = Chroma.from_documents(all_docs, embeddings, persist_directory=persistent_directory)
db.persist()

print("\nChroma vector store created and persisted successfully!")
