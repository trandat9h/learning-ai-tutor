# Overview
An AI tutor that converts passive technical content (notes, slides, web) into active learning workflows with memory, testing, and personalization.

# Structure
One single notebook `src.ipynb` contains all the code. The code can run both locally and in Google Colab. The tutor is CLI command based.

# Features Requirements
## Content management
The system manage content in a tree structure. Content is the smallest unit of the the system, representing one single concept.
User can navigate between content, content can be grouped and named as topic that group similar content in one.

## Content Ingestion 
User can upload new document in pdf or images format. The file is uploaded directly to the OpenAI Files API and passed to GPT-4o, which handles all reading and parsing natively (no local PDF/image libraries needed). The system then:
1. Analyses the content and splits it into separate concepts (fully handled by GPT-4o)
2. Adds the content into the management system
3. Ingests them into the OpenAI Vector Store for semantic search
4. Performs web search to suggest similar content that users might be interested in

## Generate questions
User can choose which topics or content user want to revise today, or the system can randomly recommend the content to test
System can generated in one of the following forms:
- An MCQ question
- An open-ended answer

Question content can be:
- Theory based: Purely based on provided content to test user understanding. Do not try to test user hard memory on every words.
- Practical based: Based on the application aspects of the content, test user's ability to recall and apply that concept into real-world example situation. Code questions are restricted to **Python or SQL only**, so they can be executed and verified automatically via the OpenAI Code Interpreter tool. If the source concept is in another language, reframe the question in Python/SQL or fall back to a theory question.

The question can be asked as the following up, which means there might be one or multiple questions can be asked in question turn


# Code conventions
- The AI utilise OpenAI API only.
- All folder can be stored offline in local file or Google Collab folder
- Each seperated logics are implemented in a single cell with markdown explaining the purpose of that cell
- All features navigation will be combined into one last block for user to run and navigate between feature using a **CLI text menu** (`print` + `input`). No GUI widgets.
- Related functions are grouped into classes (OOP) for maintainability. Each major feature area has its own class (e.g. `ContentManager`, `Ingester`, `QuestionEngine`, `ProgressTracker`).

# AI Thinking Architecture
AI is designed to think carefully before doing anything, so the process can take multiple think-action turn before the final result is given.