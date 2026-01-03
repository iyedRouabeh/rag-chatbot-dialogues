import os
import numpy as np
import psycopg
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

# -------------------------
# Config / Load env
# -------------------------
load_dotenv()  # reads src/.env too if you run from project root

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")  # fallback

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def pgvector_text(vec: np.ndarray) -> str:
    vec = np.asarray(vec, dtype=np.float32)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return "[" + ",".join(f"{float(x):.8f}" for x in vec.tolist()) + "]"

def get_conn():
    return psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource
def load_groq_client():
    if not GROQ_API_KEY:
        return None
    return Groq(api_key=GROQ_API_KEY)

def retrieve_top_k(question: str, k: int = 3):
    model = load_embedder()
    q_vec = model.encode(question, normalize_embeddings=True)
    q_txt = pgvector_text(q_vec)

    sql = """
    SELECT
      id,
      filename,
      content,
      1 - (embedding <=> (%s)::vector) AS similarity
    FROM public.dialogues
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> (%s)::vector
    LIMIT %s;
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            # IMPORTANT: if you recreate ivfflat index later, set probes here
            # cur.execute("SET ivfflat.probes = 5;")
            cur.execute(sql, (q_txt, q_txt, k))
            rows = cur.fetchall()

    # rows: (id, filename, content, similarity)
    return rows

def build_context(rows):
    # Concatenate the top dialogues as context, with simple separators
    parts = []
    for (id_, fname, content, sim) in rows:
        parts.append(f"[SOURCE id={id_} file={fname} similarity={float(sim):.3f}]\n{content}")
    return "\n\n---\n\n".join(parts)

def generate_answer(question: str, context: str):
    client = load_groq_client()
    if client is None:
        return "❌ GROQ_API_KEY manquante dans src/.env", None

    system = (
        "Tu es un assistant d'analyse de dialogues téléphoniques (hôtesse/client). "
        "Tu dois répondre UNIQUEMENT en te basant sur le CONTEXTE fourni. "
        "Si l'information n'est pas dans le contexte, dis-le clairement."
    )

    user = f"""CONTEXTE:
{context}

QUESTION:
{question}

CONSIGNE:
- Réponds en français.
- Donne une réponse structurée (points/étapes si possible).
- Termine par une mini-section: "Sources utilisées" (liste des fichiers/id)."""

    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content, resp

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="RAG Dialogues Téléphoniques", page_icon="💬", layout="wide")

st.title("💬 RAG – Analyse de Dialogues Téléphoniques")
st.caption("Recherche sémantique (pgvector) + génération (Groq) + sources")

with st.sidebar:
    st.header("⚙️ Paramètres")
    top_k = st.slider("Nombre de dialogues récupérés (Top-K)", 1, 10, 3)
    st.write("DB:", DB_NAME, "@", DB_HOST, ":", DB_PORT)
    st.write("Embedding model:", EMBED_MODEL_NAME)
    st.write("Groq model:", GROQ_MODEL)
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY manquante dans src/.env")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Pose ta question sur les dialogues :", placeholder="Ex: Comment l'hôtesse accueille un client ?")

col1, col2 = st.columns([1, 1])

if st.button("🔎 Rechercher & Répondre") and question.strip():
    rows = retrieve_top_k(question.strip(), k=top_k)

    if not rows:
        st.error("Aucun dialogue trouvé. Vérifie la DB / embeddings.")
    else:
        context = build_context(rows)
        answer, _ = generate_answer(question.strip(), context)

        st.session_state.history.append({"q": question.strip(), "a": answer, "rows": rows})

# Display chat history
for item in reversed(st.session_state.history):
    st.markdown(f"### 🧑‍💻 Question\n{item['q']}")
    st.markdown(f"### 🤖 Réponse\n{item['a']}")

    with st.expander("📌 Sources (Top-K)"):
        for (id_, fname, content, sim) in item["rows"]:
            st.markdown(f"**id={id_} | {fname} | similarity={float(sim):.3f}**")
            st.code(content[:1200])
