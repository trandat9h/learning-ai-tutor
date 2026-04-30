# AI Tutor — Implementation Plan

## Context

The project is an AI tutor that converts passive technical content (notes, slides, PDFs, images) into active learning workflows. The notebook `src.ipynb` is currently empty. The goal is to build the full system from scratch as a single Jupyter notebook that runs in both local and Google Colab environments.

**Confirmed technology choices:**
- **AI / LLM**: GPT-4o (OpenAI API) for all analysis, generation, and grading
- **Vector search**: OpenAI Vector Stores API (no separate DB needed)
- **Web search**: OpenAI Responses API with `web_search_preview` built-in tool
- **Persistence**: JSON files (works with local disk and Google Drive mount)
- **UI**: `ipywidgets` — Colab-native buttons/dropdowns
- **Personalization**: Track test history + scores + weak topics (per user, stored in JSON)

---

## Notebook Cell Structure

Each cell = one logical unit with a markdown header explaining its purpose.

### Cell 1 — Setup & Dependencies

Install and import all required libraries. Configure OpenAI API key.

```
!pip install openai ipywidgets graphviz
```

**Imports:**
- `openai` (Client, Files API, Vector Stores API, Responses API)
- `ipywidgets`, `IPython.display`
- `graphviz` — tree visualisation
- `json`, `pathlib`, `uuid`, `datetime`, `random`, `io`

Config: Set `OPENAI_API_KEY` (read from env or Colab secrets). Set `DATA_DIR` (either `/content/drive/MyDrive/ai_tutor/` for Colab or `./data/` for local).

---

### Cell 2 — Data Models & Persistence (`class Store`)

Define dataclasses for domain objects and a `Store` class that owns all JSON I/O.

**Data structures (stored as JSON):**

```
data/
  content_tree.json     # tree of Topics and Content nodes
  vector_store.json     # OpenAI vector_store_id (persisted after creation)
  user_progress.json    # test history, scores, weak topic tracking
```

**`Content` dataclass:**
```json
{
  "id": "uuid",
  "title": "string",
  "body": "string",
  "topic_id": "uuid",
  "source_file": "string",
  "vector_file_id": "string",
  "created_at": "iso8601"
}
```

**`Topic` dataclass:**
```json
{
  "id": "uuid",
  "name": "string",
  "parent_id": "uuid | null",
  "children_topic_ids": ["uuid"],
  "content_ids": ["uuid"]
}
```

**`SessionRecord` dataclass:**
```json
{
  "date": "iso8601",
  "content_id": "uuid",
  "question_type": "mcq | open",
  "correct": true,
  "score": 1.0
}
```

**`class Store`** — single source of truth for persistence:
```python
class Store:
    def __init__(self, data_dir: Path): ...
    def load(self) -> None: ...          # load all JSON files into memory
    def save(self) -> None: ...          # flush in-memory state to disk
    def get_topic(self, id) -> Topic: ...
    def get_content(self, id) -> Content: ...
    def add_topic(self, topic: Topic) -> None: ...
    def add_content(self, content: Content) -> None: ...
    def delete_content(self, id) -> None: ...
    def all_content(self) -> list[Content]: ...
    def all_topics(self) -> list[Topic]: ...
```

---

### Cell 3 — Content Management (`class ContentManager`)

Wraps the `Store` to provide tree navigation and editing operations.

```python
class ContentManager:
    def __init__(self, store: Store): ...
    def create_topic(self, name, parent_id=None) -> Topic: ...
    def add_content_to_topic(self, content: Content, topic_id) -> None: ...
    def list_topics(self, parent_id=None, indent=0) -> str: ...   # returns indented tree string
    def view_content(self, content_id) -> Content: ...
    def delete_content(self, content_id) -> str: ...              # returns vector_file_id for cleanup
    def move_content(self, content_id, new_topic_id) -> None: ...
    def render_tree(self) -> "graphviz.Digraph": ...              # inline graph in Colab/Jupyter
```

**`render_tree()` implementation:**
- Topic nodes rendered as `shape="folder"`, labelled `📁 <topic name>`
- Content nodes rendered as `shape="note"`, labelled `📄 <content title>`
- Edges flow top-down (`rankdir="TB"`): Root → subtopics → content nodes
- Returns a `graphviz.Digraph` object; calling `display(content_manager.render_tree())` renders it inline

**Dependency:** `pip install graphviz` added to Cell 1 install block.

The store initialises with a root topic named `"Root"` on first run.

---

### Cell 4 — Content Ingestion (`class Ingester`)

Delegates all file reading and concept splitting entirely to OpenAI — no local parsing libraries needed. The file is uploaded to the OpenAI Files API and passed directly to GPT-4o, which reads it natively (PDF text extraction and image vision are both handled by the model).

```python
class Ingester:
    def __init__(self, openai_client, content_manager: ContentManager,
                 vector_store: VectorStore, web_searcher: WebSearcher): ...

    def ingest(self, file_bytes: bytes, filename: str, topic_id: str) -> list[Content]:
        """Full pipeline: upload → analyse & split → store → embed → suggest."""

    def _upload_file(self, file_bytes, filename) -> str: ...
    # Uploads to OpenAI Files API, returns file_id.
    # Supported formats passed directly: PDF, PNG, JPG, WEBP, GIF.

    def _extract_concepts(self, file_id: str) -> list[dict]: ...
    # Two-step Think Architecture call with the uploaded file_id attached.
    # Returns [{"title": ..., "body": ...}]

    def _delete_upload(self, file_id: str) -> None: ...
    # Clean up the temporary Files API upload after concepts are extracted.
```

**Pipeline:**
1. **Upload** — `client.files.create(file=file_bytes, purpose="assistants")` → get `file_id`
2. **Analyse & split** — pass `file_id` to GPT-4o via the Responses API (Think Architecture two-step below)
3. **Store** — create `Content` nodes, save to `content_tree.json` via `ContentManager`
4. **Embed** — add each concept to the OpenAI Vector Store (`VectorStore.add_content`)
5. **Suggest** — run web search per concept title (`WebSearcher.suggest`)
6. **Cleanup** — delete the temporary Files API upload (`_delete_upload`)

**GPT-4o Think Architecture (two-step, file attached):**
- Step 1 (reasoning): `"You have been given a document. Identify all distinct concepts it contains. Think step by step — consider what each section teaches and how concepts relate."`
- Step 2 (extraction): `"Now output the concepts as a JSON array only, no commentary: [{\"title\": \"...\", \"body\": \"...\"}]"`

---

### Cell 5 — Vector Store (`class VectorStore`)

Wraps the OpenAI Vector Stores API.

```python
class VectorStore:
    def __init__(self, openai_client, store: Store): ...
    def get_or_create(self) -> str: ...           # returns vector_store_id, persists it
    def add_content(self, content: Content) -> str: ...   # returns file_id
    def delete_content(self, file_id: str) -> None: ...
    def search(self, query: str, top_k=5) -> list[str]: ...  # returns content_ids via file_search tool
```

---

### Cell 6 — Web Search (`class WebSearcher`)

Wraps the OpenAI Responses API `web_search_preview` tool.

```python
class WebSearcher:
    def __init__(self, openai_client): ...
    def suggest(self, concept_title: str) -> list[dict]: ...
    # Returns: [{"title": "...", "url": "...", "summary": "..."}]
```

Called automatically after each concept is ingested. Results displayed inline in the notebook.

---

### Cell 7 — Question Generation (`class QuestionEngine`)

Generates questions from a content item using a two-step Think Architecture chain.

```python
class QuestionEngine:
    def __init__(self, openai_client): ...

    def generate(self, content: Content,
                 question_type: str,    # "mcq" | "open"
                 question_style: str    # "theory" | "practical"
                 ) -> dict: ...

    def generate_followup(self, content: Content, prior_answer: str) -> dict | None: ...
    # Returns None if GPT-4o decides no follow-up is needed. Max 3 per session item.

    def _think(self, content: Content) -> str: ...    # Step 1 reasoning turn
    def _build_question(self, content, reasoning, q_type, q_style) -> dict: ...  # Step 2
```

**Code question constraint:** When `question_style == "practical"` and code is involved, only Python or SQL are used. Non-Python/SQL concepts are reframed or fall back to theory.

**MCQ output schema:**
```json
{ "question": "...", "options": ["A", "B", "C", "D"], "correct_index": 1, "explanation": "..." }
```

**Open-ended / code question output schema:**
```json
{ "question": "...", "language": "python | sql | none", "model_answer": "...", "rubric": "...", "test_code": "..." }
```

---

### Cell 8 — Answer Evaluation (`class Grader`)

```python
class Grader:
    def __init__(self, openai_client): ...

    def grade(self, question: dict, student_answer: str) -> dict: ...
    # Returns: {"score": 0.0–1.0, "feedback": "..."}

    def _grade_mcq(self, question, answer) -> dict: ...         # deterministic index compare
    def _grade_open(self, question, answer) -> dict: ...        # GPT-4o rubric grading
    def _grade_code(self, question, answer) -> dict: ...        # Code Interpreter execution
```

**Code grading flow (`_grade_code`):**
1. Merge student code + `test_code` from question schema.
2. Send to OpenAI Code Interpreter via Responses API.
3. Pass → score 1.0. Fail → error fed back to GPT-4o for a human-readable hint + style review.

After grading, `Grader` returns the result; the caller (`StudySession`) persists it via `ProgressTracker`.

---

### Cell 9 — Study Session (`class StudySession`)

Orchestrates one full study loop: question → answer → grade → (optional follow-up) → repeat.

```python
class StudySession:
    def __init__(self, content_manager: ContentManager,
                 question_engine: QuestionEngine,
                 grader: Grader,
                 progress_tracker: ProgressTracker): ...

    def start(self, content_ids: list[str],
              question_type: str, question_style: str) -> None: ...
    # Loops through content_ids, calls engine + grader, saves results, shows follow-ups.

    def _recommend_content(self) -> list[str]: ...
    # 1. weak_content_ids from progress  2. vector search on recent sessions  3. random fallback
```

---

### Cell 10 — Progress Tracking (`class ProgressTracker`)

Manages all read/write access to `user_progress.json` and computes derived stats.

```python
class ProgressTracker:
    def __init__(self, store: Store): ...
    def record(self, session_record: SessionRecord) -> None: ...
    def get_weak_content_ids(self, threshold=0.6, window=5) -> list[str]: ...
    def get_recent_sessions(self, n=10) -> list[SessionRecord]: ...
    def get_study_streak(self) -> int: ...        # consecutive days with ≥1 session
    def summary(self) -> dict: ...                # all stats for the dashboard
```

**Weak topic rule:** A content item is "weak" if its average score over the last 5 attempts is below 0.6.

---

### Cell 11 — Main Navigation UI (ipywidgets)

Final cell. Combines all features behind a tabbed interface.

**Top-level tabs** (using `ipywidgets.Tab`):
1. **Manage** — view tree, create topics, delete/move content; includes a **"View Graph"** button that calls `content_manager.render_tree()` and displays the graphviz diagram inline
2. **Ingest** — file upload widget → trigger pipeline (Cells 4–6)
3. **Study** — choose mode (manual / recommended), start session (Cells 7–9)
4. **Progress** — read-only dashboard (Cell 10)

Each tab renders its own sub-UI using `VBox`, `HBox`, `Button`, `Dropdown`, `FileUpload`, `Output`.

---

## Class Overview

| Class | Cell | Responsibility |
|-------|------|----------------|
| `Store` | 2 | JSON persistence for all domain objects |
| `ContentManager` | 3 | Topic/content tree CRUD and navigation |
| `Ingester` | 4 | PDF/image parse → concept extraction → pipeline |
| `VectorStore` | 5 | OpenAI Vector Stores API wrapper |
| `WebSearcher` | 6 | OpenAI web search suggestions |
| `QuestionEngine` | 7 | MCQ / open-ended / code question generation |
| `Grader` | 8 | Answer evaluation (deterministic, rubric, code interpreter) |
| `StudySession` | 9 | Study loop orchestration and content recommendation |
| `ProgressTracker` | 10 | Session history, weak topics, streaks |

**Dependency graph (constructor injection):**
```
Store
 ├── ContentManager(store)
 ├── VectorStore(client, store)
 └── ProgressTracker(store)

Ingester(client, content_manager, vector_store, web_searcher)
StudySession(content_manager, question_engine, grader, progress_tracker)
```

All classes are instantiated once in Cell 11 and shared across the UI tabs.

---

## Critical Files

| File | Purpose |
|------|---------|
| `src.ipynb` | The entire implementation — 11 cells |
| `data/content_tree.json` | Topic + Content nodes (auto-created on first run) |
| `data/vector_store.json` | Persisted OpenAI vector store ID |
| `data/user_progress.json` | Session history, scores, weak content IDs |

---

## Dependencies

```
openai>=1.30
ipywidgets>=8.0
graphviz>=0.20
```

All available via `pip install` in Colab. No additional API keys beyond OpenAI.

---

## Verification Plan

1. **Cell 1** — Run setup cell; confirm no import errors, API key accepted
2. **Cell 2** — Call `load_tree()` on empty state; confirm `data/` directory and default JSON files created
3. **Cell 4** — Upload a small PDF (2–3 pages); confirm concepts extracted and stored in tree
4. **Cell 5** — Confirm vector store file count increases after ingestion
5. **Cell 6** — Confirm web search results printed after ingestion
6. **Cell 7** — Generate one MCQ and one open-ended question from an ingested concept; inspect output schema
7. **Cell 8** — Submit a correct and an incorrect answer; verify scores and feedback differ
8. **Cell 9** — Run a study session from "weak topics" mode (seed one low score manually); confirm the weak topic is surfaced
9. **Cell 11** — Navigate all 4 tabs in Colab; confirm all widgets render and trigger correct backend calls
