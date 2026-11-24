<div align="center">

# 📖 Bibliobot

### Your conversational guide to *Hands-On Machine Learning*

Ask questions about the book, explore the passages behind an answer, and work through quick calculations — all from a Streamlit chat interface.

[Features](#-features) · [How it works](#-how-it-works) · [Getting started](#-getting-started) · [Project structure](#-project-structure)

</div>

---

## ✨ Features

- **Book-focused Q&A** — finds relevant passages in a local book index and answers from that context, with page references.
- **Semantic retrieval with reranking** — searches embedded text chunks, then reorders candidates with a cross-encoder for more relevant results.
- **Math and greetings** — routes simple greetings and calculation questions separately from book questions; Gemini extracts and explains calculations, while a restricted Python math evaluator computes the expression.
- **Source transparency** — expand an answer to inspect its retrieved excerpts and relevance scores.
- **Vector database inspector** — check the indexed chunk count, preview stored chunks, or look up a chunk by ID in the sidebar.
- **Local persistent index** — ChromaDB stores the book embeddings in `chroma_db/` between runs.

## 🧭 How it works

```text
PDF → extract text and page numbers → split into overlapping chunks
    → embed chunks → store in ChromaDB

Question → classify intent
    ├── Book question → retrieve candidates → rerank → Gemini answer with page references
    ├── Calculation → Gemini expression extraction → restricted math evaluation → Gemini explanation
    └── Greeting → short Gemini response
```

For book questions, the app asks the language model to answer only from the retrieved excerpts. If the index does not contain relevant material, it is instructed to say it could not find the answer in the book. Greetings and math responses follow separate paths and are not grounded in book excerpts.

## 🚀 Getting started

### 1. Requirements

- Python and `pip`
- A Google Gemini API key
- A local, legally obtained copy of the book PDF for building the index

The PDF is not included in this repository. The first run may download the embedding and reranker models used by Sentence Transformers, so allow time and disk space for those model files.

### 2. Install dependencies

From the repository root, create and activate a virtual environment, then install the pinned packages:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 3. Configure Gemini

Set `GOOGLE_API_KEY` in the same terminal session that will run the app. Do not commit the key to the repository.

```bash
# macOS / Linux
export GOOGLE_API_KEY="your-api-key"

# Windows PowerShell
$env:GOOGLE_API_KEY = "your-api-key"
```

### 4. Build the book index

Place your PDF in the project (for example, under `book/`) and update the PDF path in the `__main__` section of `embed.py` to point to that file. The script currently contains a Windows-style example path, so adjust it when running on macOS or Linux.

Then build the local ChromaDB index once:

```bash
python embed.py
```

This extracts page-aware text, creates overlapping chunks, embeds them with `all-MiniLM-L6-v2`, and writes them to `chroma_db/`. The chat app expects this index to exist and contain chunks before it is launched.

### 5. Launch Bibliobot

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit. Ask a question about the indexed book, try a calculation, or use the sidebar to inspect stored chunks. Retrieved passages for book answers are available in the **Retrieved chunks** expander.

## 🗂️ Project structure

| File or directory | Purpose |
| --- | --- |
| `app.py` | Streamlit chat UI, session history, source excerpts, and vector database inspector. |
| `generate.py` | Intent routing and Gemini-powered answer generation. |
| `retrieve.py` | ChromaDB access, semantic retrieval, and cross-encoder reranking. |
| `embed.py` | Sentence Transformer embeddings and vector index creation. |
| `chunker.py` | Splits extracted text into overlapping chunks while retaining page metadata. |
| `load_book.py` | Extracts text and page mappings from a PDF with `pypdf`. |
| `math_agent.py` | Evaluates mathematical expressions using a restricted set of built-ins and `math` functions. |
| `chroma_db/` | Persistent local ChromaDB data created by the indexing step. |
| `requirements.txt` | Pinned Python dependencies. |

## ⚙️ Models and configuration

The current implementation uses:

- **Answer generation and intent-related text:** `gemini-3.5-flash-lite` through the Google GenAI SDK.
- **Text embeddings:** `all-MiniLM-L6-v2` through Sentence Transformers.
- **Passage reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Vector storage:** ChromaDB, using the `book_chunks` collection in `chroma_db/`.

Model names, chunk sizing, and retrieval counts are currently configured in the Python source rather than through environment variables. `GOOGLE_API_KEY` is the environment variable required by the Gemini client.

## 🛠️ Troubleshooting

| Problem | What to check |
| --- | --- |
| The app cannot find book chunks | Run `python embed.py` first and confirm `chroma_db/` contains the `book_chunks` collection. |
| PDF loading fails | Confirm the PDF exists at the path configured in `embed.py`; update the sample Windows-style path for your operating system. |
| Gemini requests fail | Confirm `GOOGLE_API_KEY` is set in the terminal running Streamlit and that the key has access to the configured model. |
| First launch is slow | Sentence Transformers may need to download the embedding and reranker models on first use. |

## 🔒 Privacy and limitations

The source PDF and ChromaDB index are local inputs, but prompts and retrieved text used for generated responses are sent to the configured Gemini service. Do not index or submit material unless you have permission to use it with that service. Answers depend on the PDF text extraction and indexed chunks; scanned pages without extractable text may not be searchable. The app is designed around the book index created for this project, not as a general-purpose document manager.

## 🤝 Contributing

Bug reports and focused improvements are welcome. When changing the retrieval or indexing pipeline, please keep page metadata intact so book answers can continue to point readers to their sources.
