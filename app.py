import uuid
import streamlit as st
from datetime import datetime
from prototype2 import ask_vaultify
from guardrails import apply_guardrails, validate_output


# ============================================================
# PAGE CONFIG
# ============================================================
# Colors, fonts and dark mode all come from .streamlit/config.toml
# (native Streamlit theming) rather than injected CSS — this is the
# safest way to restyle Streamlit and can't produce a blank page.

st.set_page_config(
    page_title="Vaultify AI Support",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# THEME POLISH — one static, self-contained CSS block
# ============================================================
# Colors/dark-mode/font come from .streamlit/config.toml (native
# theming). This adds only what config.toml can't express: gradient
# text, glow, and glass blur. It's a single unbroken <style> tag with
# no dynamic content, so it can't produce the broken-HTML issue that
# custom chat bubbles caused earlier.

st.markdown(
    """
    <style>
    h1 {
        font-weight: 800 !important;
        letter-spacing: -1px;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 6px !important;
        border-color: #262626 !important;
    }

    [data-testid="stChatMessage"] {
        border-radius: 6px;
    }

    button[kind="primary"] {
        background: #FF5C38 !important;
        border: none !important;
        border-radius: 6px !important;
        color: #0B0B0C !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }

    button[kind="secondary"] {
        border-radius: 6px !important;
    }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:8].upper()


def add_message(role, content):
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "time": datetime.now().strftime("%H:%M")
    })


# ============================================================
# SIDEBAR — native containers, no custom HTML
# ============================================================

with st.sidebar:

    col_logo, col_name = st.columns([1, 4])
    with col_logo:
        st.markdown("### 🔐")
    with col_name:
        st.markdown("**Vaultify**")
        st.caption("AI SUPPORT CONSOLE")

    if st.button("＋ New conversation", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.session_id = uuid.uuid4().hex[:8].upper()
        st.rerun()

    st.divider()

    st.caption("SYSTEM")
    with st.container(border=True):
        c1, c2 = st.columns([2, 1])
        c1.markdown("AI Support Engine")
        c2.markdown(":green[●] **Online**")

    st.caption("KNOWLEDGE BASE")
    with st.container(border=True):
        st.markdown("📄 **Product Documentation**")
        st.caption("Features · Plans · Security")
    with st.container(border=True):
        st.markdown("🗂️ **Issue Knowledge Base**")
        st.caption("Historical resolutions")

    st.caption("CAPABILITIES")
    st.markdown(
        "- Product information\n"
        "- Technical troubleshooting\n"
        "- Error code assistance\n"
        "- Enterprise & compliance\n"
        "- Security & administration"
    )

    st.divider()
    st.caption("Vaultify Support Platform  \nEnterprise AI Assistant · v1.0")


# ============================================================
# MAIN HEADER
# ============================================================

col_title, col_session = st.columns([4, 1])

with col_title:
    #st.markdown(":orange-background[**● AI SUPPORT ASSISTANT**]")
    st.title("How can we help?")
    #st.markdown(
    #    "Ask about Vaultify features, backup and sync, security, "
    #    "enterprise administration, and APIs — or describe a technical "
    #    "issue and get an answer sourced from product documentation and "
    #    "past resolutions."
    #)
    b1, b2, b3 = st.columns(3)
    b1.caption("🕐 24/7 availability")
    b2.caption("🔎 Multi-source retrieval")
    b3.caption("🛡️ Enterprise-grade security")

#with col_session:
#    st.caption(f"SESSION\n\n`{st.session_state.session_id}`")

#st.write("")


# ============================================================
# SUGGESTION VARIABLES
# ============================================================

question1 = question2 = question3 = question4 = False


# ============================================================
# WELCOME SCREEN
# ============================================================

if len(st.session_state.messages) == 0:

    with st.container(border=True):
        st.markdown("**Welcome to Vaultify Support**")
        st.write(
            "I'm your AI-powered support assistant. Ask about features, "
            "backup and sync, security, enterprise administration, or "
            "APIs — or describe a technical issue you're running into "
            "and I'll pull the closest answer from documentation and "
            "prior resolutions."
        )

    st.caption("TRY ASKING")

    col1, col2 = st.columns(2, gap="small")

    with col1:
        question1 = st.button(
            "📦 Backup vs Sync — what's the difference?",
            use_container_width=True, key="suggestion_1"
        )
        question3 = st.button(
            "⚠️ My 30GB file isn't syncing — how can I fix it?",
            use_container_width=True, key="suggestion_3"
        )

    with col2:
        question2 = st.button(
            "🛡️ What security features does Enterprise provide?",
            use_container_width=True, key="suggestion_2"
        )
        question4 = st.button(
            "🔧 I'm getting ERR-SY161. What should I do?",
            use_container_width=True, key="suggestion_4"
        )


# ============================================================
# DISPLAY CHAT HISTORY — native st.chat_message, always renders
# ============================================================

for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🔐"
    with st.chat_message(message["role"], avatar=avatar):
        st.caption(message.get("time", ""))
        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

typed_query = st.chat_input("Ask about Vaultify or describe your issue...")
#st.caption("Vaultify AI can make mistakes. Verify critical details in official documentation.")


# ============================================================
# SUGGESTED QUERY
# ============================================================

suggested_query = None
if question1:
    suggested_query = "What is the difference between Backup and Sync?"
elif question2:
    suggested_query = "What security features does Vaultify Enterprise provide?"
elif question3:
    suggested_query = "My 30GB file isn't syncing. How can I fix it?"
elif question4:
    suggested_query = "I'm getting ERR-SY161. What should I do?"

user_query = typed_query or suggested_query


# ============================================================
# PROCESS QUERY
# ============================================================

if user_query:

    add_message("user", user_query)

    with st.chat_message("user", avatar="🧑‍💻"):
        st.caption(st.session_state.messages[-1]["time"])
        st.markdown(user_query)


    # Apply input guardrails
    guardrail_result = apply_guardrails(user_query)


    with st.chat_message("assistant", avatar="🔐"):

        # Query blocked by guardrails
        if not guardrail_result["allowed"]:

            final_response = guardrail_result["message"]

        # Query is safe -> Send to RAG
        else:

            with st.spinner("Searching Vaultify knowledge..."):

                try:
                    final_response = ask_vaultify(user_query)

                    # Apply output guardrails
                    final_response = validate_output(final_response)

                except Exception:
                    final_response = (
                        "I couldn't process your request right now. "
                        "Please try again in a moment."
                    )


        st.caption(datetime.now().strftime("%H:%M"))
        st.markdown(final_response)


    add_message("assistant", final_response)


    if suggested_query:
        st.rerun()