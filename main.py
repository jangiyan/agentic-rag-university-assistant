import os
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
load_dotenv()
api_key = os.getenv("api_key")
model_name = os.getenv("model")
client = Groq(
    api_key=api_key
)
pdf_path = "university_handbook_sample.pdf"
reader = PdfReader(pdf_path)
text = ""
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_text(text)
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_db = Chroma.from_texts(
    texts=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

@tool
def search_university_documents(query: str):
    """
    Search university documents and return relevant information.
    """

    results = vector_db.similarity_search(
        query,
        k=3
    )

    context = "\n\n".join(
        result.page_content
        for result in results
    )

    return context
class AgentState(TypedDict):
    question: str
    decision: str
    answer: str
def agent(state: AgentState):

    question = state["question"]

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": """
You are an AI routing agent.

Decide whether the user's question requires information
from university documents.

If it requires information from university documents,
respond only with:

RAG

Otherwise respond only with:

DIRECT
"""
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    decision = response.choices[0].message.content.strip().upper()

    return {
        "decision": decision
    }

def rag_node(state: AgentState):

    question = state["question"]

    # Tool call
    context = search_university_documents.invoke(
        {
            "query": question
        }
    )

    # Generate answer
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": """
You are a university assistant.

Answer the user's question using ONLY the provided context.

If the answer is not available in the context,
say:

I could not find this information in the university documents.
"""
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{question}
"""
            }
        ]
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer
    }
def direct_node(state: AgentState):

    question = state["question"]

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": question
            }
        ]
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer
    }
def route_question(state: AgentState):

    if state["decision"] == "RAG":
        return "rag"

    return "direct"
graph = StateGraph(AgentState)


# Add Nodes
graph.add_node(
    "agent",
    agent
)

graph.add_node(
    "rag",
    rag_node
)

graph.add_node(
    "direct",
    direct_node
)


# Start → Agent
graph.add_edge(
    START,
    "agent"
)

graph.add_conditional_edges(
    "agent",
    route_question,
    {
        "rag": "rag",
        "direct": "direct"
    }
)


# End
graph.add_edge(
    "rag",
    END
)

graph.add_edge(
    "direct",
    END
)
app = graph.compile()
