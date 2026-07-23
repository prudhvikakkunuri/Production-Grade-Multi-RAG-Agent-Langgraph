import os
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
# HYBRID SEARCH & CROSS-ENCODER IMPORTS
# ============================================================
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Please add it to your .env file."
    )

# ============================================================
# FILE PATHS
# ============================================================

PRODUCT_PDF_PATH = "data/vaultify_product_manual.pdf"
LOGS_CSV_PATH = "data/logs.csv"

# ============================================================
# LLM
# ============================================================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    api_key=GROQ_API_KEY
)

# ============================================================
# EMBEDDING & RERANKER MODELS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize Cross-Encoder model for reranking candidates
cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")


# ============================================================
# PRODUCT RETRIEVER (HYBRID SEARCH + RERANKER)
# ============================================================

def create_product_retriever():
    loader = PyMuPDFLoader(PRODUCT_PDF_PATH)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    product_chunks = text_splitter.split_documents(documents)

    # 1. Sparse / Lexical Retriever (BM25 - fetches candidate pool of 8)
    bm25_retriever = BM25Retriever.from_documents(product_chunks)
    bm25_retriever.k = 8

    # 2. Dense / Semantic Retriever (FAISS - fetches candidate pool of 8)
    vectorstore = FAISS.from_documents(
        documents=product_chunks,
        embedding=embeddings
    )
    dense_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 8}
    )

    # 3. Hybrid Ensemble Retriever (RRF combination)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.4, 0.6]
    )

    # 4. Cross-Encoder Reranker (narrows candidates down to top 4)
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=4)

    # 5. Combined Compression Pipeline
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )


# ============================================================
# ISSUE LOG RETRIEVER (HYBRID SEARCH + RERANKER)
# ============================================================

def create_issue_retriever():
    loader = CSVLoader(
        file_path=LOGS_CSV_PATH,
        encoding="utf-8"
    )
    issue_documents = loader.load()

    # 1. Sparse / Lexical Retriever (BM25 - key for exact error code & log ID matching)
    bm25_retriever = BM25Retriever.from_documents(issue_documents)
    bm25_retriever.k = 6

    # 2. Dense / Semantic Retriever (FAISS)
    vectorstore = FAISS.from_documents(
        documents=issue_documents,
        embedding=embeddings
    )
    dense_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 6}
    )

    # 3. Hybrid Ensemble Retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.5, 0.5]
    )

    # 4. Cross-Encoder Reranker (narrows candidates down to top 3)
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)

    # 5. Combined Compression Pipeline
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )


# ============================================================
# INITIALIZE RETRIEVERS
# ============================================================

product_retriever = create_product_retriever()
issue_retriever = create_issue_retriever()


# ============================================================
# LANGGRAPH STATE
# ============================================================
class State(TypedDict):
    messages: Annotated[
        list,
        add_messages
    ]
    query_type: str
    retrieved_context: str


# ============================================================
# QUERY CLASSIFIER
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
1. PRODUCT
Use "product" when the user is asking for information about
Vaultify or how a Vaultify feature works.
Examples:
- What is Vaultify?
- What are the features of Vaultify?
- How does Backup work?
- What is the difference between Backup and Sync?
- What plans are available?
- How much does Pro cost?
- Does Vaultify support SSO?
- What encryption does Vaultify use?
- What is CMEK?
- Does Vaultify provide API access?
- What is the API rate limit?
- What compliance certifications does Vaultify have?
- What data residency options are available?
- How does Litigation Hold work?
- What can an administrator do?
- How do I enable a Vaultify feature?

2. ISSUE
Use "issue" when the customer is experiencing a problem
or needs troubleshooting assistance.
Examples:
- My backup isn't working.
- My backup is stuck at 0%.
- My files aren't syncing.
- My 30GB video won't sync.
- I cannot log in.
- My API isn't working.
- I'm getting a 429 error.
- Slack notifications aren't arriving.
- My audit export is stuck.
- I cannot share a file.
- ERR-SY161
- ERR-BK101

Any error code or troubleshooting request should normally
be classified as "issue".

3. GENERAL
Use "general" for:
- Greetings
- Thank-you messages
- Casual conversation
- Questions unrelated to Vaultify

IMPORTANT:
Return ONLY one word:
product
OR
issue
OR
general

USER QUERY:
{user_query}
"""
    response = llm.invoke(
        classifier_prompt
    )
    query_type = (
        response.content
        .strip()
        .lower()
    )
    # Safety fallback in case the classifier
    # returns unexpected text.
    if query_type not in [
        "product",
        "issue",
        "general"
    ]:
        query_type = "general"

    return {"query_type": query_type}


# ============================================================
# QUERY ROUTER
# ============================================================

def route_query(state: State) -> Literal["product_rag","issue_rag","general"]:
    query_type = state["query_type"]
    if query_type == "product":
        return "product_rag"
    elif query_type == "issue":
        return "issue_rag"
    return "general"


# ============================================================
# PRODUCT RAG
# ============================================================

def product_rag_node(state: State):
    user_query = state["messages"][-1].content
    product_docs = product_retriever.invoke(
        user_query
    )
    product_context = "\n\n".join(
        [
            f"""
            PRODUCT DOCUMENT:
            {doc.page_content}"""
            for doc in product_docs
        ]
    )
    return {
        "retrieved_context": product_context
    }


# ============================================================
# ISSUE RAG
# ============================================================

def issue_rag_node(state: State):
    user_query = state["messages"][-1].content
    # --------------------------------------------------------
    # RETRIEVE HISTORICAL ISSUES
    # --------------------------------------------------------
    issue_docs = issue_retriever.invoke(
        user_query
    )
    issue_context = "\n\n".join(
        [
            f"""
HISTORICAL ISSUE:
{doc.page_content}
"""
            for doc in issue_docs
        ]
    )
    # --------------------------------------------------------
    # RETRIEVE PRODUCT INFORMATION
    # --------------------------------------------------------
    product_docs = product_retriever.invoke(
        user_query
    )
    product_context = "\n\n".join(
        [
            f"""
PRODUCT DOCUMENTATION:
{doc.page_content}
"""
            for doc in product_docs
        ]
    )
    # --------------------------------------------------------
    # COMBINE BOTH SOURCES
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
    return {"retrieved_context": combined_context}


# ============================================================
# PRODUCT RESPONSE
# ============================================================

def product_response_node(state: State):

    user_query = state["messages"][-1].content

    context = state["retrieved_context"]

    system_prompt = """
You are Vaultify Support AI, an intelligent customer support
assistant for the Vaultify cloud backup and file synchronization
SaaS platform.

Your job is to answer product-related questions using the
retrieved Vaultify product documentation.


RULES:

1. Base Vaultify-specific information on the supplied product
   documentation.

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

3. Explain the answer clearly and professionally.

4. When the customer asks how to perform an action,
   provide clear step-by-step instructions.

5. Keep the response easy for customers to understand.

6. Do not discuss internal implementation concepts such as:

   - RAG
   - FAISS
   - Embeddings
   - Vector databases
   - Retrieved chunks

7. If the supplied documentation does not contain enough
   information to answer confidently, clearly state that
   the available Vaultify documentation does not provide
   enough information.
"""

    user_prompt = f"""
VAULTIFY PRODUCT DOCUMENTATION:

{context}


CUSTOMER QUESTION:

{user_query}


Answer the customer's question using the supplied
Vaultify documentation.
"""

    response = llm.invoke(
        [
            SystemMessage(
                content=system_prompt
            ),

            HumanMessage(
                content=user_prompt
            )
        ]
    )

    return {
        "messages": [
            AIMessage(
                content=response.content
            )
        ]
    }


# ============================================================
# ISSUE RESPONSE
# ============================================================

def issue_response_node(state: State):
    user_query = state["messages"][-1].content
    context = state["retrieved_context"]
    system_prompt = """
You are Vaultify Support AI, an intelligent technical support
assistant for the Vaultify cloud backup and file synchronization
SaaS platform.

You troubleshoot customer problems using:

1. Historical Vaultify issue/resolution records.
2. Vaultify product documentation.


TROUBLESHOOTING RULES:

1. Analyze the customer's symptoms carefully.

2. Use historical issue records to identify relevant
   previously resolved problems.

3. Use the product documentation to support troubleshooting
   instructions and explain Vaultify product behavior.

4. If the customer provides an error code and the SAME
   error code appears in the retrieved support knowledge,
   prioritize that record.

5. When a reliable historical match exists, mention:

   - Log ID
   - Error Code
   - Product Area
   - Recommended Resolution

6. NEVER invent:

   - Log IDs
   - Error codes
   - Product settings
   - Product features
   - Troubleshooting procedures
   - Product limits

7. If a retrieved historical issue is only SIMILAR to the
   customer's symptoms, do not claim that it is definitely
   the same problem.

   Say:

   "This appears similar to known issue KB-XXXX."

8. If there is no reliable historical match, clearly say
   that no confirmed historical issue was found.

9. If the issue cannot be reliably resolved from the
   available information, recommend checking:

   Settings > Diagnostics

   for an error code.

10. Recommend contacting Vaultify Support when the supplied
    knowledge does not provide a reliable resolution.

11. Whenever possible, provide troubleshooting instructions
    as numbered steps.

12. Do not expose implementation details such as RAG,
    FAISS, embeddings, or vector retrieval.
"""

    user_prompt = f"""
VAULTIFY SUPPORT KNOWLEDGE:

{context}


CUSTOMER ISSUE:

{user_query}


Analyze the issue and provide the safest and most relevant
troubleshooting response based on the supplied knowledge.
"""

    response = llm.invoke(
        [
            SystemMessage(
                content=system_prompt
            ),

            HumanMessage(
                content=user_prompt
            )
        ]
    )

    return {
        "messages": [
            AIMessage(
                content=response.content
            )
        ]
    }


# ============================================================
# GENERAL RESPONSE
# ============================================================

def general_node(state: State):
    user_query = state["messages"][-1].content
    system_prompt = """
You are Vaultify Support AI.

Vaultify is a cloud backup, file synchronization,
file sharing, and collaboration SaaS platform.

For greetings and casual conversation, respond naturally
and professionally.

If the user asks something unrelated to Vaultify, politely
explain that you are designed primarily to help with
Vaultify product information and customer support.

Keep general responses concise.
"""

    response = llm.invoke(
        [
            SystemMessage(
                content=system_prompt
            ),

            HumanMessage(
                content=user_query
            )
        ]
    )

    return {
        "messages": [
            AIMessage(
                content=response.content
            )
        ]
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

builder = StateGraph(State)


# ============================================================
# ADD NODES
# ============================================================

builder.add_node("classifier", classifier_node)

builder.add_node("product_rag", product_rag_node)

builder.add_node("issue_rag", issue_rag_node)

builder.add_node("product_response", product_response_node)

builder.add_node("issue_response", issue_response_node)

builder.add_node("general", general_node)


# ============================================================
# START
# ============================================================

builder.add_edge(START, "classifier")


# ============================================================
# CONDITIONAL ROUTING
# ============================================================

builder.add_conditional_edges(
    "classifier",
    route_query,
    {
        "product_rag": "product_rag",
        "issue_rag": "issue_rag",
        "general": "general"
    }
)


# ============================================================
# PRODUCT PATH
# ============================================================

builder.add_edge("product_rag", "product_response")

builder.add_edge("product_response", END)


# ============================================================
# ISSUE PATH
# ============================================================

builder.add_edge("issue_rag", "issue_response")

builder.add_edge("issue_response", END)


# ============================================================
# GENERAL PATH
# ============================================================

builder.add_edge("general", END)


# ============================================================
# COMPILE GRAPH
# ============================================================

graph = builder.compile()


# ============================================================
# FUNCTION USED BY STREAMLIT
# ============================================================

def ask_vaultify(user_query: str):
    """
    This is the only function the Streamlit UI needs to call.

    It sends the user's query through the complete
    Vaultify LangGraph workflow and returns the final response.
    """

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
    return result["messages"][-1].content
