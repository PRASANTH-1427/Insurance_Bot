"""
Streamlit Chat Frontend for InsureAssist AI
============================================

A conversational chat interface that connects to the InsureAssist AI FastAPI backend
and maintains full conversation history.

Usage:
    streamlit run frontend-streamlit/file/app.py
"""

import os
import uuid
import requests
import streamlit as st

# Configure page layout and metadata
st.set_page_config(
    page_title="InsureAssist AI",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------
# Sidebar Configuration & Controls
# ---------------------------------------------------------
with st.sidebar:
    st.title("🛡️ InsureAssist AI")
    st.caption("AI-Powered Multi-Agent Insurance Assistant")

    st.divider()

    # Backend API configuration
    api_url = st.text_input(
        "API Chat Endpoint",
        value=os.getenv("INSUREASSIST_API_URL", "http://localhost:8000/api/v1/chat"),
        help="FastAPI endpoint for chat messages.",
    )

    # Liveness check button
    health_url = api_url.replace("/chat", "/health")
    if st.button("Check Backend Status", use_container_width=True):
        try:
            res = requests.get(health_url, timeout=3)
            if res.status_code == 200:
                st.success("Backend is online and healthy! ✅")
            else:
                st.warning(f"Backend responded with status code: {res.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Cannot connect to backend: {e}")

    st.divider()

    # Session management
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    st.write(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
    st.caption(f"Full ID: `{st.session_state.session_id}`")

    # New Conversation / Reset button
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------
# Session State Initialization (Message History)
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 I'm your InsureAssist AI assistant.\n\n"
                "I can help you explore insurance policies, generate quotes, or assist you with lead intake.\n\n"
                "How can I help you today?"
            ),
            "metadata": None,
        }
    ]

# ---------------------------------------------------------
# Chat Display
# ---------------------------------------------------------
st.header("💬 InsureAssist AI Chat")
st.caption("Ask questions about policies, get insurance recommendations, or register your details.")

# Render existing conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("metadata"):
            meta = msg["metadata"]
            active_agent = meta.get("active_agent")
            intent = meta.get("intent")
            reason = meta.get("reason")
            if active_agent or intent or reason:
                with st.expander("🔍 Agent Details", expanded=False):
                    if active_agent:
                        st.write(f"**Active Agent:** `{active_agent}`")
                    if intent:
                        st.write(f"**Intent:** `{intent}`")
                    if reason:
                        st.write(f"**Routing Reason:** {reason}")

# ---------------------------------------------------------
# Chat Input & Message Processing
# ---------------------------------------------------------
if prompt := st.chat_input("Type your message here..."):
    # 1. Append and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Call the FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("InsureAssist is thinking..."):
            payload = {
                "message": prompt,
                "session_id": st.session_state.session_id,
            }

            try:
                response = requests.post(
                    api_url,
                    json=payload,
                    timeout=60,
                )

                if response.status_code == 200:
                    data = response.json()
                    reply_text = data.get("reply", "No reply received.")
                    metadata = {
                        "active_agent": data.get("active_agent"),
                        "intent": data.get("intent"),
                        "reason": data.get("reason"),
                    }

                    # Display assistant response
                    st.markdown(reply_text)

                    # Display routing metadata if present
                    if any(metadata.values()):
                        with st.expander("🔍 Agent Details", expanded=False):
                            if metadata.get("active_agent"):
                                st.write(f"**Active Agent:** `{metadata['active_agent']}`")
                            if metadata.get("intent"):
                                st.write(f"**Intent:** `{metadata['intent']}`")
                            if metadata.get("reason"):
                                st.write(f"**Routing Reason:** {metadata['reason']}")

                    # Append to session state history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": reply_text,
                            "metadata": metadata,
                        }
                    )
                else:
                    error_msg = f"⚠️ Server Error ({response.status_code}): {response.text}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg, "metadata": None}
                    )

            except requests.exceptions.ConnectionError:
                error_msg = (
                    "⚠️ **Could not connect to the FastAPI backend.**\n\n"
                    f"Please make sure the API is running at `{api_url}`:\n\n"
                    "```bash\n"
                    "uvicorn app.api.main:app --reload\n"
                    "```"
                )
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg, "metadata": None}
                )
            except Exception as e:
                error_msg = f"⚠️ Unexpected error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg, "metadata": None}
                )
