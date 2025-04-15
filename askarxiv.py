# 🔍 AskArXiv: RAG with Mistral + Cloud SQL + Deployment (Colab-ready)

# ── 2. Import Libraries ─────────────────────────────────────────────
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, Table, Column, String, MetaData
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
import torch
import pandas as pd
import gradio as gr
from sqlalchemy.sql import text
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── 3. Fetch arXiv Metadata via API ────────────────────────────────────
# Using the `arxiv` Python package to pull cs.LG papers
import arxiv

search = arxiv.Search(
    query="cat:cs.LG",
    max_results=100,
    sort_by=arxiv.SortCriterion.SubmittedDate
)
ml_papers = []
for result in search.results():
    ml_papers.append({
        "id": result.entry_id.split('/')[-1],
        "title": result.title,
        "abstract": result.summary
    })

print(ml_papers[:10])
# Convert to list of dicts
# ml_papers now holds up to 1000 ML papers

# ── 4. Create SQL Table & Store Abstracts ─────────────────────────── Create SQL Table & Store Abstracts ─────────────────────────── & Store Abstracts ─────────────────────────── Create SQL Table & Store Abstracts ───────────────────────────

engine = create_engine("sqlite:///arxiv.db", connect_args={"check_same_thread": False, "timeout": 30})



# Enable Write-Ahead Logging for better concurrency
with engine.connect() as conn:
    conn.execute(text("PRAGMA journal_mode=WAL"))

metadata = MetaData()
papers_table = Table(
    "papers", metadata,
    Column("id", String, primary_key=True),
    Column("title", String),
    Column("abstract", String),
    Column("embedding", String),  # serialized embedding
)

metadata.create_all(engine)


model_embed = SentenceTransformer("all-MiniLM-L6-v2")

conn = engine.connect()
# Store in DB

for paper in ml_papers:
    try:
        emb = model_embed.encode(paper["abstract"]).tolist()
        conn.execute(papers_table.insert().values(
            id=paper["id"],
            title=paper["title"],
            abstract=paper["abstract"],
            embedding=','.join(map(str, emb))
        ))
        conn.commit()
    except Exception as e:
        print(f"❌ Skipping {paper['id']} - Error: {e}")
            
result = conn.execute(text("SELECT COUNT(*) FROM papers"))
print("Inserted documents:", result.scalar())


result = conn.execute(text("SELECT * FROM papers LIMIT 1"))
paper_sample = result.fetchone()
print(paper_sample)

# ── 5. Define Search & RAG Functions ───────────────────────────────


def get_top_k_docs(query, k=5):
    q_embed = model_embed.encode(query)
    df = pd.read_sql("SELECT * FROM papers", conn)
    if df.empty:
        print("⚠️ No documents found in DB.")
        return pd.DataFrame(columns=["id", "title", "abstract", "embedding"])
    try:
        embeds = np.vstack(df["embedding"].apply(lambda x: list(map(float, x.split(",")))))
    except ValueError:
        print("⚠️ Invalid embeddings found — possibly null/empty strings.")
        return pd.DataFrame(columns=["id", "title", "abstract", "embedding"])
    sims = cosine_similarity([q_embed], embeds)[0]
    top_idx = sims.argsort()[-k:][::-1]
    return df.iloc[top_idx]


# ── 6. Load Mistral Model ───────────────────────────────────────────
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# ── 7. Generate Answer with Mistral ─────────────────────────────────
def generate_answer_mistral(query):
    docs = get_top_k_docs(query, k=5)
    context = "\n\n".join(docs["abstract"])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    #inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Truncate input properly to avoid overflow
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,  # match model's max token length
        padding="max_length"
    ).to(model.device)

    outputs = model.generate(**inputs, max_length=2048, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)



# Load tokenizer and add pad token
model_name = "tiiuae/falcon-rw-1b"
device = "cuda" if torch.cuda.is_available() else "cpu"

gpt2_tokenizer = AutoTokenizer.from_pretrained(model_name)
gpt2_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token


# ✅ Resize model embeddings *after* adding tokens to tokenizer
gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))

# ✅ Move model to correct device (same as above)
# gpt2_model = gpt2_model.to("cuda" if torch.cuda.is_available() else "cpu")




def generate_answer_gpt2(query):
    docs = get_top_k_docs(query, k=5)
    context = "\n\n".join(docs["abstract"])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    inputs = gpt2_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding="max_length"
    )  # ✅ Use correct device

    outputs = gpt2_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=100,
        pad_token_id=gpt2_tokenizer.pad_token_id
    )
    
    return gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)



# ── 9. Evaluation ───────────────────────────────────────────────────
from scipy.stats import ttest_rel

queries = [
    "What are recent methods for improving graph neural networks?",
    "How is contrastive learning used in NLP?",
    "Explain self-supervised learning in computer vision.",
    "How does reinforcement learning work for recommendation systems?",
    "What is the role of attention in transformers?"
]

mistral_outputs = [generate_answer_mistral(q) for q in queries]
gpt2_outputs = [generate_answer_gpt2(q) for q in queries]

def score_response(resp, query):
    q_emb = model_embed.encode(query)
    r_emb = model_embed.encode(resp)
    return np.dot(q_emb, r_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(r_emb))

mistral_scores = [score_response(r, q) for r, q in zip(mistral_outputs, queries)]
gpt2_scores = [score_response(r, q) for r, q in zip(gpt2_outputs, queries)]

t_stat, p_val = ttest_rel(mistral_scores, gpt2_scores)
print("Mistral Scores:", mistral_scores)
print("GPT-2 Scores:", gpt2_scores)
print(f"Paired t-test result: t={t_stat:.3f}, p={p_val:.4f}")

import gradio as gr

# ── Gradio Deployment ──────────────────────────────────────────
def rag_interface(query, model_choice):
    if model_choice == "Mistral":
        return generate_answer_mistral(query)
    else:
        return generate_answer_gpt2(query)

gr.Interface(
    fn=rag_interface,
    inputs=[
        gr.Textbox(label="Your Question"),
        gr.Radio(["Mistral", "GPT-2"], label="Choose Model")
    ],
    outputs="text",
    title="AskArXiv RAG Search",
).launch(share=True)

