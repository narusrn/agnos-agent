# Agnos-Agent: Health Consultation Chatbot

A **Streamlit-based chatbot** that assists users with health-related questions using **LangGraph workflows**, retrieval from historical Q&A, and a GPT-4o-powered LLM. The system ensures responses are grounded in documents and avoids hallucinations.

Try it here: [Agnos-Agent Live Demo](https://agnos-agent-hbdghnp5uwfrgeojdmfxhy.streamlit.app)
---

## Features

- Interactive chat interface via **Streamlit**.
- Handles user input with **context-aware LLM** responses.
- Retrieval from a **document-based forum dataset** (`data.xlsx`).
- **Retry logic** for failed responses.
- Uses **workflow nodes**:
  - `document_search`: retrieve documents from the knowledge base.
  - `generate`: produce candidate answers.
  - `transform_query`: rephrase questions to improve retrieval.
  - `finalize_response`: return the final answer.
- **Grading mechanism** ensures answers are grounded and relevant.
- Thai-language support.

---

## Installation

1. Clone the repo:

```bash
git clone https://github.com/narusrn/agnos-agent.git
cd agnos-agent
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your **OpenAI API key** (example in `app.py`):

```python
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
```

---

## Project Structure

```
.
├── app.py                  # Streamlit app entry point
├── src/
│   ├── data/         # Core services: Agent, callback 
│       └── data.xlsx       # Forum Q&A dataset
│   ├── services/         # Core services: Agent, callback 
│       └── agent.py       # 
│       └── callbacks.py       # 
│       └── tools.py       # 
│       └── workflow.py       # 
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

- Enter your health-related question in Thai.
- The agent will respond based on previous Q&A data and GPT-4o generation.
- Messages are stored in session state to maintain chat history.

---

## Agent Workflow

1. **Document Retrieval** (`document_search`)  
   Fetches relevant documents based on user query.

2. **Answer Generation** (`generate`)  
   Produces a candidate answer using GPT-4o and RAG prompt chaining.

3. **Query Transformation** (`transform_query`)  
   Improves queries if the answer does not address the question.

4. **Finalization** (`finalize_response`)  
   Returns the verified answer to the user.

5. **Grading**  
   - `GradeHallucinations`: checks if answer is grounded in retrieved documents.
   - `GradeAnswer`: checks if the answer addresses the user's question.

