# Frontend Streamlit Chat Interface

This folder contains a simple chat interface built with [Streamlit](https://streamlit.io/) for interactive testing of the InsureAssist AI multi-agent system.

## Folder Structure

```
frontend-streamlit/
└── file/
    └── app.py       # Main Streamlit chat application
```

## How to Run

1. **Ensure dependencies are installed**:
   ```bash
   pip install streamlit requests
   ```
   *(or `pip install -e .` from the project root)*

2. **Ensure your FastAPI backend is running**:
   ```bash
   uvicorn app.api.main:app --reload
   ```

3. **Start the Streamlit application**:
   ```bash
   streamlit run frontend-streamlit/file/app.py
   ```

4. The browser will open automatically at `http://localhost:8501`.

## Features
- **Persistent Message History**: Retains the entire chat history in `st.session_state`.
- **Session ID Tracking**: Connects turns together so the multi-agent graph remembers lead details and conversation context.
- **Agent Transparency**: Expandable metadata showing which agent answered (`lead_agent`, `insurance_agent`, etc.), detected intent, and reasoning.
- **New Conversation Button**: Instantly resets conversation history and initializes a new session.
