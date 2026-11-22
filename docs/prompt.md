# Resume Reader

Build a Resume Reader that accepts one PDF resume and answers questions using only the contents of that resume, with retrieval-augmented generation. Keep it simple. Keep the frontend and the API as separate apps.

## What it does

- Upload a PDF, extract the text, chunk it by resume section, embed the chunks locally, and store them in Postgres with pgvector under a new document id.
- Then chat over multiple turns.
- Each turn normalizes the question, retrieves the nearest chunks of that document, and asks an LLM to answer in markdown from those excerpts only, using earlier turns to resolve follow-ups.
- Return each source with its page, section when known, and excerpt.
- A new upload starts a new conversation.
- If the excerpts do not contain the answer, reply exactly: `That information is not in the resume.`

## Stack

- Backend: Python 3.12, uv, FastAPI, Uvicorn, PyMuPDF, Sentence Transformers, pgvector on Postgres 16, Groq over httpx, Pydantic.
- Frontend: React, TypeScript, Vite, pnpm.
- Run Postgres from Docker Compose with `pgvector/pgvector:pg16`, database `resume_reader`, user and password `postgres`, port 5432.
- The API listens on 8000 and the frontend dev server on 5173.
- The frontend only calls the HTTP API.

## PDF and chunking

- Accept a real PDF only, and reject files over 10 MB by reading the body in bounded pieces.
- Extract text per page with PyMuPDF: 1-based pages, drop blank pages, and on a two-column page read the left column then the right, keeping full-width lines in vertical order.
- An unreadable PDF is 400.
- A PDF with no extractable text is 422.
- Treat a line as a heading when, ignoring case and a trailing colon, it is summary, profile, objective, experience, employment, education, skills, strengths, projects, certifications, or contributions, optionally preceded by professional, core, technical, selected, key, or relevant.
- Carry the section name onto later lines and later pages until the next heading.
- Text before the first heading is still chunked, with no section.
- Split a section's text on the page by paragraphs, then lines, then sentences, then words, then characters, at 800 characters with 120 overlap.
- A short section on a page stays one chunk.
- Each chunk keeps its index, page, section, and text.

## Embeddings, search, and questions

- Embed with `all-MiniLM-L6-v2` on CPU: 384-dimensional unit-length vectors, loaded once, with parsing and embedding off the event loop.
- Every upload is a new document.
- Search is exact cosine distance within that document, nearest first, top-k 5, with no approximate index.
- Collapse whitespace in the question and reject a blank one; do not otherwise rewrite it.
- The client sends prior turns and the server does not store the chat.
- Retrieve with the latest question plus the previous user question.
- Give the model prior turns only to resolve what the user is referring to.

## Generation

- Keep the prompt separate from the provider call.
- The model may use only the excerpts, must not use outside knowledge, writes markdown, and cites excerpt number, page, and section.
- Skip the model when nothing is retrieved and return the not-in-resume sentence with no sources.
- Call Groq through a small provider-neutral async interface.
- Default model is `llama-3.3-70b-versatile`.
- A missing API key fails at startup.
- Provider failures are HTTP 502.
- Never log the key or the resume.

## HTTP

- `GET /health` is healthy only when the database answers, otherwise 503.
- `POST /documents` takes multipart field `file` and returns the id, filename, page count, chunk count, and status `indexed`.
- `POST /query` takes `document_id`, `question`, and prior turns, and returns `answer` and `sources` in excerpt order.
- An unknown document is 404.
- A blank question is 422.

## Settings

- Load settings from the environment and reject impossible values at startup: embedding dimension other than 384, non-positive limits, overlap that is negative or not smaller than the chunk size, and a non-positive top-k.
- Ship an example env file with an empty Groq key.
- The default CORS origin is `http://localhost:5173`.
- The frontend reads `VITE_API_URL`, defaulting to `http://localhost:8000`.

## Engineering

- Separate ingestion, chunking, embeddings, storage, the LLM provider, query normalization, retrieval, generation, and prompt text from the HTTP layer.
- Use type hints and Pydantic at the boundary.
- Apply the schema on startup.
- Log ingestion, retrieval, and generation without logging the resume.
- No extra frameworks, queues, auth, or services.

## Frontend

- One screen: a PDF upload, then a multi-turn transcript.
- Render answers as markdown (paragraphs, lists, emphasis, and headings) and leave the user's question as plain text.
- Rendered markdown must not execute HTML.
- Show citations apart from the answer, and link excerpt-number markers to the matching source.
- Include clear loading, empty, and error states, and keep the layout usable on a narrow viewport.
- No component library and no RAG logic in the client.
- Check behavior with the PDFs in `corpus/`.
