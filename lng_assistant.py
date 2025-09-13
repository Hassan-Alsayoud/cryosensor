import os
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import shutil
from pathlib import Path
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# langchain / integrations
try:
    from langchain_groq import ChatGroq
except Exception as e:
    raise ImportError(
        "langchain_groq not installed or not importable. Install: pip install langchain-groq"
    ) from e

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.chains import RetrievalQA
except Exception as e:
    raise ImportError("langchain core APIs not available. Install/upgrade langchain.") from e

# Chroma vector store (preferred here because your earlier script used it)
try:
    from langchain_chroma import Chroma
except Exception as e:
    raise ImportError(
        "langchain_chroma not installed. Install: pip install langchain-chroma\n"
        "Also make sure chromadb (or chroma dependencies) are installed."
    ) from e

# Embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception as e:
    raise ImportError("langchain_huggingface not installed. Install: pip install langchain-huggingface") from e

# PDF loader and text splitter
try:
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception as e:
    raise ImportError(
        "Missing document loader or text splitter. Install: pip install langchain-community langchain-text-splitter"
    ) from e

# Thermodynamics library: use `thermo`.
try:
    from thermo import Mixture
except Exception as e:
    raise ImportError(
        "thermo package not found. Install it with: pip install thermo (or use CoolProp if you prefer)"
    ) from e

# ----------------- Configuration -----------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set. Export it before running.")

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
DOCS_DIR = os.environ.get("DOCS_DIR", "docs")
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "rag_chroma")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# FastAPI app
app = FastAPI(title="LNG Assistant (RAG + Groq + thermo)")

# allow basic CORS for testing if needed
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve the static site from ./static (index.html will be served at /)
from fastapi.responses import FileResponse

@app.get("/", include_in_schema=False)
def root():
    # serve your SPA entrypoint
    return FileResponse(Path("static/index.html"))

# serve other assets (css/js/images) from /static/...
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------- Vector DB + Embeddings --------------
embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vectordb: Optional[Chroma] = None

# Attempt to load persisted Chroma DB if present
if Path(CHROMA_PERSIST_DIR).exists():
    try:
        vectordb = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embedding_fn)
    except Exception:
        vectordb = None

# If no persisted DB, build from PDFs in DOCS_DIR
if vectordb is None:
    docs_path = Path(DOCS_DIR)
    if not docs_path.exists():
        raise RuntimeError(f"No vector DB found at '{CHROMA_PERSIST_DIR}' and docs directory '{DOCS_DIR}' not found.")

    loader = PyPDFDirectoryLoader(DOCS_DIR)
    docs = loader.load()
    if not docs:
        raise RuntimeError(f"No documents loaded from {DOCS_DIR}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    # create and persist
    vectordb = Chroma.from_documents(splits, embedding_fn, persist_directory=CHROMA_PERSIST_DIR)

retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# -------------- LLM (Groq) --------------
llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY)

# Define a small Answer TypedDict for attempted structured output
class AnswerSchema(TypedDict, total=False):
    summary: str
    key_points: List[str]
    sources: List[str]

# Build prompt (keeps instructions tight to encourage JSON)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer only using the provided context. "
               "When possible, output strict JSON matching the schema: "
               "{{'summary': str, 'key_points': [str], 'sources': [str]}}"),
    ("human", "Question:\n{question}\n\nContext:\n{context}")
])

# -------------- Thermo helper --------------

def compute_thermo_props_from_composition(composition: Dict[str, float], T: float, P: float) -> Dict[str, Any]:
    """
    composition: mapping of component name -> mole fraction (sum should be ~1.0)
    T: temperature in Kelvin
    P: pressure in Pascal

    Returns a dict with keys: phase, density (kg/m^3), enthalpy (J/mol), entropy (J/mol/K)
    """
    # Normalize fractions
    names = list(composition.keys())
    zs = list(composition.values())
    total = sum(zs)
    if total <= 0:
        raise ValueError("Composition fractions must sum to > 0")
    zs = [z / total for z in zs]

    # thermo.Mixture expects lists of names and mole fractions; provide T and P
    # NOTE: thermo.Mixture uses T in K and P in Pa.
    mix = Mixture(IDs=names, zs=zs, T=T, P=P)

    # Access attributes; attribute names may vary across versions — try common ones
    props: Dict[str, Any] = {}
    props["phase"] = getattr(mix, "phase", None)
    props["density"] = getattr(mix, "rho", None)  # kg/m^3
    props["enthalpy"] = getattr(mix, "H", None)
    props["entropy"] = getattr(mix, "S", None)

    return props

# -------------- API schemas --------------
class QueryRequest(BaseModel):
    query: str
    process_data: Dict[str, Any]  # expected keys: composition (dict), temperature (K), pressure (Pa)


@app.post("/ask")
async def handle_query(req: QueryRequest):
    # validate process_data
    pd = req.process_data
    if not all(k in pd for k in ("composition", "temperature", "pressure")):
        raise HTTPException(status_code=400, detail="process_data must include 'composition', 'temperature', and 'pressure'")

    composition = pd["composition"]
    T = float(pd["temperature"])  # expect Kelvin
    P = float(pd["pressure"])     # expect Pascal

    try:
        thermo = compute_thermo_props_from_composition(composition, T, P)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Thermo computation failed: {e}")

    # Build context from retrieved docs
    retrieved = retriever.invoke(req.query)
    if not retrieved:
        # proceed with just thermo enrichment
        context = f"Thermodynamic state:\n{thermo}"
    else:
        context = "\n\n".join(
            f"[{Path(d.metadata.get('source','')).name} p.{d.metadata.get('page','?')}]\n{d.page_content}"
            for d in retrieved
        )
        context = context + "\n\nThermodynamic state:\n" + str(thermo)

    msgs = prompt.invoke({"question": req.query, "context": context})

    # Try structured output first (if model supports it); else fall back to plain text
    try:
        structured_llm = llm.with_structured_output(AnswerSchema)  # may raise NotImplementedError
        structured_result = structured_llm.invoke(msgs)
        return {"answer": structured_result, "thermo": thermo}
    except Exception:
        # fallback to raw text
        try:
            raw = llm.invoke(msgs)
            # attempt to parse JSON from start of the text if present
            import json
            s = raw.content.strip() if hasattr(raw, "content") else str(raw).strip()
            if s.startswith("{"):
                try:
                    parsed = json.loads(s)
                    return {"answer": parsed, "thermo": thermo}
                except Exception:
                    # return raw text if parsing fails
                    return {"answer": {"summary": raw}, "thermo": thermo}
            else:
                return {"answer": {"summary": raw}, "thermo": thermo}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM invocation failed: {e}")


# Health check endpoint
@app.get("/health")
async def health():
    return {
        "ok": True,
        "model": GROQ_MODEL,
        "chroma_persist": str(Path(CHROMA_PERSIST_DIR).exists()),
    }

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        dst_dir = Path(DOCS_DIR)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / file.filename

        # save the uploaded file
        with open(dst_path, "wb") as f:
            f.write(await file.read())

        # load & split
        docs = _load_docs_from_dir(dst_dir) if dst_path.suffix.lower() != ".pdf" else PyPDFLoader(str(dst_path)).load()
        splits = _split_docs(docs if isinstance(docs, list) else docs)

        # init/create the vector DB if needed
        global vectordb
        if vectordb is None:
            vectordb = Chroma(
                collection_name="docs",
                embedding_function=embedding_fn,
                persist_directory=CHROMA_PERSIST_DIR,
            )

        # add and persist
        vectordb.add_documents(splits)
        vectordb.persist()

        return {"filename": file.filename, "indexed_chunks": len(splits)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload/index failed: {e}")

@app.post("/reindex")
def reindex():
    try:
        # rebuild clean
        if Path(CHROMA_PERSIST_DIR).exists():
            shutil.rmtree(CHROMA_PERSIST_DIR)

        docs_dir = Path(DOCS_DIR)
        docs_dir.mkdir(parents=True, exist_ok=True)

        docs = _load_docs_from_dir(docs_dir)
        splits = _split_docs(docs)

        global vectordb
        vectordb = Chroma(
            collection_name="docs",
            embedding_function=embedding_fn,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        if splits:
            vectordb.add_documents(splits)
            vectordb.persist()

        return {"indexed_chunks": len(splits)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reindex failed: {e}")
	
