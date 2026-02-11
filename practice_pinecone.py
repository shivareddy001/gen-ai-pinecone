import os
from tqdm import tqdm
import uuid
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# ---------------- CONFIG ----------------

PDF_FOLDER = r"C:\Users\shiva\Desktop\AI Best match code\pinecone\Education"     # folder containing PDFs
INDEX_NAME = "pdf-search"
PINECONE_REGION = "us-east-1"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ----------------------------------- Initialize embedding model ------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = model.get_sentence_embedding_dimension()
print("Embedding dimension:", embedding_dim)

# -------------------------- Initialize Pinecone --------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)


# ------------------------ Create index if not exists ----------------------------------
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=embedding_dim,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
    )
    print("Index created successfully")
else:
    print("Index already exists")

index = pc.Index(INDEX_NAME)

# ----------- Read PDFs -------------

documents = []

for file in os.listdir(PDF_FOLDER):
    if file.endswith(".pdf"):
        path = os.path.join(PDF_FOLDER, file)
        reader = PdfReader(path)

        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""

        documents.append((file, text))

print(f"Loaded {len(documents)} PDFs")
# --------------------- GENERATE EMBEDDINGS ---------------------

texts = [doc[1] for doc in documents]
ids = [str(i) for i in range(len(documents))]
metadatas = [{"source": doc[0]} for doc in documents]

print("Generating embeddings...")
embeddings = model.encode(texts, batch_size=16, show_progress_bar=True).tolist()


# -------------------------------- UPSERT TO PINECONE --------------------------------

print("Uploading vectors to Pinecone...")
BATCH_SIZE = 100

for i in range(0, len(ids), BATCH_SIZE):
    batch_vectors = []
    for j in range(i, min(i + BATCH_SIZE, len(ids))):
        batch_vectors.append((ids[j], embeddings[j], metadatas[j]))

    index.upsert(vectors=batch_vectors)
    print(f"Uploaded batch {(i//BATCH_SIZE) + 1}")

print("All PDFs uploaded successfully!")

# ----------- Upload to Pinecone -------------

vectors = []

for i, (filename, text) in enumerate(tqdm(documents)):
    embedding = model.encode(text[:4000]).tolist()  # limit text size
    vectors.append(
        (
            str(i),
            embedding,
            {"filename": filename}
        )
    )

index.upsert(vectors)

print("Upload complete.")

# ----------- Search Function -------------

def search_pdfs(query, top_k=5):
    query_embedding = model.encode(query).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    print("\nTop Matches:\n")

    for match in results["matches"]:
        print(f"PDF: {match['metadata']['filename']}")
        print(f"Score: {match['score']}")
        print("------")


# ----------- Example Query -------------

user_query = input("Enter your search query: ")

search_pdfs(user_query)

