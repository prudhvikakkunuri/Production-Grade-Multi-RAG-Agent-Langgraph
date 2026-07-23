import os
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from typing import TypedDict, Annotated, Literal


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please add it to your .env file.")
    st.stop()


# ============================================================
# FILE PATHS
# ============================================================

PRODUCT_PDF_PATH = "data/vaultify_product_manual.pdf"
LOGS_CSV_PATH = "data\logs.csv"


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    api_key=GROQ_API_KEY
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

embeddings = embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2" )



# ============================================================
# LOAD PRODUCT KNOWLEDGE BASE
# ============================================================

@st.cache_resource
def create_product_retriever():

    # Load product PDF
    loader = PyMuPDFLoader(PRODUCT_PDF_PATH)

    documents = loader.load()

    # Product information is large, so we chunk it
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    product_chunks = text_splitter.split_documents(documents)

    # Create FAISS vector database
    vectorstore = FAISS.from_documents(
        documents=product_chunks,
        embedding=embeddings
    )

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4
        }
    )

    return retriever


# ============================================================
# LOAD ISSUE / RESOLUTION LOGS
# ============================================================

@st.cache_resource
def create_issue_retriever():

    # Each CSV row becomes a separate Document
    loader = CSVLoader(
        file_path=LOGS_CSV_PATH,
        encoding="utf-8"
    )

    issue_documents = loader.load()

    # IMPORTANT:
    # Do NOT chunk the issue logs.
    #
    # Each row already contains:
    #
    # log_id
    # product_area
    # error_code
    # issue
    # resolution
    #
    # Therefore each row should remain one complete document.

    vectorstore = FAISS.from_documents(
        documents=issue_documents,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 3
        }
    )

    return retriever


# ============================================================
# INITIALIZE RETRIEVERS
# ============================================================

product_retriever = create_product_retriever()

issue_retriever = create_issue_retriever()


# ============================================================
# LANGGRAPH STATE
# ============================================================

class State(TypedDict):

    messages: Annotated[list, add_messages]

    query_type: str

    retrieved_context: str


# ============================================================
# QUERY CLASSIFIER NODE
# ============================================================

def classifier_node(state: State):

    user_query = state["messages"][-1].content

    classifier_prompt = f"""
        You are a query classifier for Vaultify Support AI.

        Vaultify is a cloud backup, file synchronization, file sharing,
        collaboration, and enterprise file-management SaaS platform.

        Classify the user's query into exactly ONE of these categories:

        product
        issue
        general


        CATEGORY DEFINITIONS:

        1. product

        Use "product" when the user is asking for information about Vaultify,
        including:

        - What Vaultify is
        - Product features
        - How Vaultify works
        - Backup
        - Backup History
        - Sync
        - File sharing
        - Team folders
        - Permissions
        - Storage
        - Pricing
        - Plans
        - Billing information
        - Mobile application
        - Security
        - Encryption
        - 2FA
        - SSO
        - CMEK
        - Integrations
        - API
        - Data residency
        - Compliance
        - Admin Console
        - Audit logs
        - Retention
        - Support plans
        - SLA
        - Product specifications
        - System requirements
        - How to use a Vaultify feature


        2. issue

        Use "issue" when the customer is experiencing a problem
        or needs troubleshooting help.

        Examples include:

        - Something is not working
        - Backup failed
        - Backup is stuck
        - Sync is not working
        - File cannot be synchronized
        - Sharing problem
        - Login problem
        - Billing problem
        - API problem
        - Integration problem
        - Mobile application problem
        - Error message
        - Error code
        - ERR-SY161
        - ERR-BK101
        - KB issue
        - Customer asking why something failed
        - Customer asking how to fix something


        3. general

        Use "general" only for:

        - Greetings
        - Casual conversation
        - Thank you messages
        - Questions unrelated to Vaultify


        IMPORTANT:

        Return ONLY one word.

        product

        OR

        issue

        OR

        general


        USER QUERY:

        {user_query}
        """
    response = llm.invoke(classifier_prompt)
    query_type = response.content.strip().lower()

    # Fallback
    if query_type not in ["product", "issue", "general"]:
        query_type = "general"

    return {
        "query_type": query_type
    }


# ============================================================
# ROUTER FUNCTION
# ============================================================

def route_query(
    state: State,
) -> Literal["product_rag", "issue_rag", "general"]:

    query_type = state["query_type"]

    if query_type == "product":
        return "product_rag"

    elif query_type == "issue":
        return "issue_rag"

    else:
        return "general"


# ============================================================
# PRODUCT RAG NODE
# ============================================================

def product_rag_node(state: State):
    user_query = state["messages"][-1].content
    # Search only product documentation
    product_docs = product_retriever.invoke(user_query)

    product_context = "\n\n".join(
        [ f"""PRODUCT DOCUMENT:{doc.page_content}""" for doc in product_docs ]
    )

    return {
        "retrieved_context": product_context
    }


# ============================================================
# ISSUE RAG NODE
# ============================================================

def issue_rag_node(state: State):

    user_query = state["messages"][-1].content

    # --------------------------------------------------------
    # Search historical issue logs
    # --------------------------------------------------------

    issue_docs = issue_retriever.invoke(user_query)

    issue_context = "\n\n".join(
        [ f"""HISTORICAL ISSUE:{doc.page_content}""" for doc in issue_docs ]
    )


    # --------------------------------------------------------
    # Search product documentation
    # --------------------------------------------------------

    product_docs = product_retriever.invoke(user_query)

    product_context = "\n\n".join(
        [ f"""PRODUCT DOCUMENTATION:{doc.page_content}""" for doc in product_docs ]
    )
    # --------------------------------------------------------
    # Combine both knowledge sources
    # --------------------------------------------------------

    combined_context = f"""
============================================================
HISTORICAL ISSUE / RESOLUTION KNOWLEDGE
============================================================
{issue_context}
============================================================
VAULTIFY PRODUCT DOCUMENTATION
============================================================
{product_context}
"""
    return {
        "retrieved_context": combined_context
    }


# ============================================================
# PRODUCT RESPONSE NODE
# ============================================================

def product_response_node(state: State):

    user_query = state["messages"][-1].content

    context = state["retrieved_context"]

    system_prompt = """
You are Vaultify Support AI, an intelligent customer support
assistant for the fictional Vaultify cloud backup and file
synchronization SaaS platform.

Your job is to answer product-related questions using the
retrieved Vaultify product documentation.

RULES:

1. Base all Vaultify-specific information on the retrieved context.

2. Do not invent:
   - Product features
   - Pricing
   - Storage limits
   - Security capabilities
   - Product specifications
   - Settings
   - Plan benefits
   - API limits
   - Compliance certifications
   - Support SLAs

3. If the retrieved documentation contains the answer,
   explain it clearly and directly.

4. Give step-by-step instructions when the customer asks
   how to use or configure a feature.

5. Keep the response customer-friendly and professional.

6. Do not mention technical RAG concepts such as:
   - vector database
   - embeddings
   - FAISS
   - retrieved chunks
   - retrieval pipeline

7. If the answer cannot be determined from the provided
   documentation, clearly say that the available Vaultify
   documentation does not contain enough information.

8. Do not pretend that Vaultify is a real company.
   Treat it as the product represented by the supplied
   knowledge base.
"""

    user_prompt = f"""
VAULTIFY PRODUCT DOCUMENTATION:

{context}


CUSTOMER QUESTION:

{user_query}


Answer the customer's question using the documentation above.
"""

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
    )

    return {
        "messages": [
            AIMessage(content=response.content)
        ]
    }


# ============================================================
# ISSUE RESPONSE NODE
# ============================================================

def issue_response_node(state: State):

    user_query = state["messages"][-1].content

    context = state["retrieved_context"]

    system_prompt = """
You are Vaultify Support AI, an intelligent technical support
assistant for the Vaultify cloud backup and file synchronization
SaaS platform.

You help customers troubleshoot problems using:

1. Historical Vaultify issue/resolution logs.
2. Official Vaultify product documentation.


IMPORTANT TROUBLESHOOTING RULES:

1. Use historical issue logs to identify previously resolved
   problems that are relevant to the customer's symptoms.

2. Use product documentation to support the troubleshooting
   instructions and explain relevant Vaultify behavior.

3. If the customer provides an ERROR CODE and an exact matching
   error code appears in the retrieved context, prioritize that
   historical issue.

4. When there is a clear matching historical issue, mention:

   - Log ID
   - Error Code
   - Product Area
   - Recommended Resolution

5. NEVER invent:

   - Log IDs
   - Error codes
   - Product settings
   - Product features
   - Troubleshooting steps
   - Storage limits
   - Product behavior

6. If a historical issue is only SIMILAR to the customer's
   problem, do NOT claim that it is definitely the same issue.

   Instead say something such as:

   "This appears similar to known issue KB-XXXX."

7. If there is no reliable matching issue in the supplied
   knowledge, explain that no confirmed historical match
   was found.

8. If the issue cannot be reliably resolved using the supplied
   information, recommend checking:

   Settings > Diagnostics

   for an error code and contacting Vaultify Support.

9. Give troubleshooting instructions as clear numbered steps
   whenever possible.

10. Do not expose technical implementation details such as
    embeddings, FAISS, vector retrieval, or RAG.
"""

    user_prompt = f"""
SUPPORT KNOWLEDGE:

{context}


CUSTOMER ISSUE:

{user_query}


Analyze the customer's problem using the historical issue
knowledge and Vaultify product documentation.

Provide the safest and most relevant troubleshooting response.
"""

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
    )

    return {
        "messages": [
            AIMessage(content=response.content)
        ]
    }


# ============================================================
# GENERAL NODE
# ============================================================

def general_node(state: State):

    user_query = state["messages"][-1].content

    system_prompt = """
You are Vaultify Support AI.

Vaultify is a fictional cloud backup, file synchronization,
file sharing, and collaboration SaaS platform.

For greetings and casual conversation, respond naturally
and professionally.

If the user asks something unrelated to Vaultify, politely
explain that you are primarily designed to help with
Vaultify product information and customer support.

Keep general responses concise.
"""

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query)
        ]
    )

    return {
        "messages": [
            AIMessage(content=response.content)
        ]
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

builder = StateGraph(State)


# Add nodes
builder.add_node(
    "classifier",
    classifier_node
)

builder.add_node(
    "product_rag",
    product_rag_node
)

builder.add_node(
    "issue_rag",
    issue_rag_node
)

builder.add_node(
    "product_response",
    product_response_node
)

builder.add_node(
    "issue_response",
    issue_response_node
)

builder.add_node(
    "general",
    general_node
)


# ============================================================
# GRAPH EDGES
# ============================================================

builder.add_edge(
    START,
    "classifier"
)


builder.add_conditional_edges(
    "classifier",
    route_query,
    {
        "product_rag": "product_rag",
        "issue_rag": "issue_rag",
        "general": "general"
    }
)


# Product path
builder.add_edge(
    "product_rag",
    "product_response"
)

builder.add_edge(
    "product_response",
    END
)


# Issue path
builder.add_edge(
    "issue_rag",
    "issue_response"
)

builder.add_edge(
    "issue_response",
    END
)


# General path
builder.add_edge(
    "general",
    END
)


# ============================================================
# COMPILE GRAPH
# ============================================================

graph = builder.compile()


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Vaultify Support AI",
    page_icon="☁️",
    layout="centered"
)


# ============================================================
# STREAMLIT TITLE
# ============================================================

st.title("☁️ Vaultify Support AI")

st.caption(
    "AI-powered product information and intelligent "
    "customer troubleshooting assistant"
)


# ============================================================
# INITIALIZE CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I'm Vaultify Support AI. "
                "I can help you understand Vaultify features, "
                "plans, security, backup, sync, sharing, APIs, "
                "and troubleshoot product issues. "
                "How can I help you today?"
            )
        }
    ]


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# USER INPUT
# ============================================================

user_query = st.chat_input(
    "Ask about Vaultify or describe your issue..."
)


# ============================================================
# PROCESS QUERY
# ============================================================

if user_query:

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)


    # --------------------------------------------------------
    # Run LangGraph
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Analyzing your request..."):

            try:

                result = graph.invoke(
                    {
                        "messages": [
                            HumanMessage(
                                content=user_query
                            )
                        ],
                        "query_type": "",
                        "retrieved_context": ""
                    }
                )

                final_response = result["messages"][-1].content

                st.markdown(final_response)


            except Exception as e:

                final_response = (
                    "Sorry, I encountered an error while processing "
                    "your request. Please try again."
                )

                st.error(final_response)

                # During development you can uncomment this:
                # st.exception(e)


    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": final_response
        }
    )