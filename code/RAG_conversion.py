import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# 1) Load environment variables (e.g. OPEN_API_KEY)
load_dotenv()
openai_api_key = os.getenv('OPEN_API_KEY')
if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables. Please set OPEN_API_KEY in your .env file.")

# 2) Folder containing your product text files
folder_name = "company_products"
current_dir = os.getcwd()
folder_path = os.path.join(current_dir,'..//', folder_name)

# 3) Directory where Chroma will persist the database
#    (feel free to rename "chroma_db" to something else or place it anywhere else)
persistent_directory = os.path.join(current_dir,'..//', "chroma_db")

# 4) Find all .txt files in the folder
file_paths = []
for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):
        file_paths.append(os.path.join(folder_path, file_name))

if not file_paths:
    raise ValueError(f"No .txt files found in '{folder_path}'.")

# 5) Initialize a text splitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# 6) Load documents and split them
all_docs = []
for path in file_paths:
    print(f"Loading file: {path}")
    loader = TextLoader(path, encoding="utf-8")
    documents = loader.load()
    # Each 'documents' is typically a list of Document objects. Split them:
    split_docs = text_splitter.split_documents(documents)
    all_docs.extend(split_docs)

print(f"\nTotal chunks across all files: {len(all_docs)}")

# 7) Create embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api_key)

# 8) Create a Chroma vector store from the documents
print("\nCreating Chroma vector store...")
db = Chroma.from_documents(all_docs, embeddings, persist_directory=persistent_directory)

# 9) Persist the database to disk
db.persist()
print("\nChroma vector store created and persisted successfully!")

