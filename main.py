import os
import requests

from typing import TypedDict

from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END

load_dotenv()

api_key = os.getenv("api_key")
model_name = os.getenv("model")

adzuna_app_id = os.getenv("ADZUNA_APP_ID")
adzuna_app_key = os.getenv("ADZUNA_APP_KEY")

client = Groq(
    api_key=api_key
)
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
vector_db = None
full_cv_text = ""
def extract_cv_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    cv_text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            cv_text += page_text

    return cv_text


def chunk_cv_text(cv_text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_text(
        cv_text
    )

    return chunks
def create_cv_vector_db(chunks):

    vector_database = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model
    )

    return vector_database
def process_cv(uploaded_file):

    global vector_db
    global full_cv_text

    # Purani CV memory clear
    vector_db = None
    full_cv_text = ""

    # New CV extract
    cv_text = extract_cv_text(uploaded_file)

    if not cv_text.strip():

        return {
            "success": False,
            "message": "Could not extract text from the CV."
        }

    # New CV save
    full_cv_text = cv_text

    # New chunks
    chunks = chunk_cv_text(cv_text)

    # New Vector DB
    vector_db = create_cv_vector_db(chunks)

    return {
        "success": True,
        "message": "New CV processed successfully!",
        "chunks": len(chunks)
    }
@tool
def search_cv(query: str):
    """
    Search the uploaded CV and return relevant
    information based on the user's query.
    """

    if vector_db is None:

        return "No CV has been uploaded yet."

    results = vector_db.similarity_search(
        query,
        k=3
    )

    context = "\n\n".join(
        result.page_content
        for result in results
    )

    return context

@tool
def search_jobs(query: str):
    """
    Search for relevant jobs using the Adzuna API.
    """

    url = (
        "https://api.adzuna.com/"
        "v1/api/jobs/us/search/1"
    )

    params = {
        "app_id": adzuna_app_id,
        "app_key": adzuna_app_key,
        "what": query,
        "results_per_page": 10
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

    except requests.RequestException as error:

        return {
            "error": f"Job search failed: {str(error)}",
            "jobs": []
        }


    jobs = []

    for job in data.get("results", []):

        jobs.append({

            "title": job.get("title"),

            "company": job.get(
                "company",
                {}
            ).get("display_name"),

            "location": job.get(
                "location",
                {}
            ).get("display_name"),

            "description": job.get(
                "description"
            ),

            "link": job.get(
                "redirect_url"
            )
        })

    return jobs

class AgentState(TypedDict):

    user_request: str
    decision: str
    cv_context: str
    job_query: str
    job_results: list
    answer: str
def agent_router(state: AgentState):

    user_request = state["user_request"]

    response = client.chat.completions.create(

        model=model_name,

        messages=[

            {
                "role": "system",

                "content": """
You are an AI routing agent.

Decide whether the user's request requires
searching for real job opportunities.

If the user wants to:

- find jobs
- search jobs
- recommend jobs
- find suitable jobs
- find job opportunities
- discover job openings

respond ONLY with:

JOB_SEARCH

Otherwise respond ONLY with:

DIRECT
"""
            },

            {
                "role": "user",
                "content": user_request
            }

        ]
    )

    decision = (
        response
        .choices[0]
        .message
        .content
        .strip()
        .upper()
    )

    # Safety fallback
    if decision not in [
        "JOB_SEARCH",
        "DIRECT"
    ]:

        decision = "DIRECT"

    return {
        "decision": decision
    }

def route_request(state: AgentState):

    if state["decision"] == "JOB_SEARCH":

        return "job_search"

    return "direct"

def cv_search_node(state: AgentState):

    user_request = state["user_request"]

    cv_context = search_cv.invoke(
        {
            "query": user_request
        }
    )

    return {
        "cv_context": cv_context
    }

def job_query_node(state: AgentState):

    global full_cv_text

    user_request = state["user_request"]

    response = client.chat.completions.create(

        model=model_name,

        messages=[

            {
                "role": "system",

                "content": """
You are a CV analysis and job search query generator.

Analyze the candidate's CV carefully.

Determine the candidate's actual professional background,
skills, education, projects, and experience.

Generate ONE job title that best matches the candidate's CV.

IMPORTANT RULES:

- Do NOT always return Data Scientist.
- Do NOT choose Data Scientist unless the CV clearly shows
  data science, machine learning, Python data analysis,
  statistics, or related experience.

Examples:

If CV contains:
Python, Pandas, Machine Learning
→ Data Scientist

If CV contains:
JavaScript, React, Node.js
→ Software Engineer

If CV contains:
FastAPI, APIs, Python backend
→ Backend Developer

If CV contains:
SQL, Excel, Tableau, dashboards
→ Data Analyst

If CV contains:
Network security, penetration testing
→ Cybersecurity Analyst

Return ONLY ONE job title.

No explanation.
No sentence.
No extra text.
"""
            },

            {
                "role": "user",

                "content": f"""
User Request:

{user_request}

FULL CANDIDATE CV:

{full_cv_text}
"""
            }

        ]
    )

    job_query = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

    print("GENERATED JOB QUERY:", job_query)

    return {
        "job_query": job_query
    }

def job_search_node(state: AgentState):

    job_query = state["job_query"]

    job_results = search_jobs.invoke(
        {
            "query": job_query
        }
    )

    return {
        "job_results": job_results
    }

def direct_node(state: AgentState):

    user_request = state["user_request"]

    response = client.chat.completions.create(

        model=model_name,

        messages=[

            {
                "role": "system",

                "content": """
You are a helpful AI career assistant.

Answer clearly and concisely.
"""
            },

            {
                "role": "user",
                "content": user_request
            }

        ]
    )

    answer = (
        response
        .choices[0]
        .message
        .content
    )

    return {
        "answer": answer
    }

def final_job_answer_node(state: AgentState):

    user_request = state["user_request"]

    cv_context = state["cv_context"]

    job_query = state["job_query"]

    job_results = state["job_results"]


    # Check API error
    if isinstance(job_results, dict):

        if "error" in job_results:

            return {
                "answer": job_results["error"]
            }


    # Check if no jobs found
    if not job_results:

        return {
            "answer": (
                f"No jobs were found for '{job_query}'. "
                "Try another related job title."
            )
        }


    response = client.chat.completions.create(

        model=model_name,

        messages=[

            {
                "role": "system",

                "content": """
You are an AI job recommendation assistant.

Use ONLY the provided real job search results.

Recommend the most suitable jobs based on
the candidate's CV.

For each recommendation include:

- Job Title
- Company
- Location
- Why it matches the candidate
- Job Link

Do not invent job opportunities.
"""
            },

            {
                "role": "user",

                "content": f"""
User Request:
{user_request}

Generated Job Search Query:
{job_query}

Relevant Candidate CV Information:
{cv_context}

Real Job Search Results:
{job_results}
"""
            }

        ]
    )

    answer = (
        response
        .choices[0]
        .message
        .content
    )

    return {
        "answer": answer
    }


graph = StateGraph(
    AgentState
)

graph.add_node(
    "agent",
    agent_router
)

graph.add_node(
    "cv_search",
    cv_search_node
)

graph.add_node(
    "job_query",
    job_query_node
)

graph.add_node(
    "job_search",
    job_search_node
)

graph.add_node(
    "direct",
    direct_node
)

graph.add_node(
    "final_answer",
    final_job_answer_node
)

graph.add_edge(
    START,
    "agent"
)

graph.add_conditional_edges(

    "agent",

    route_request,

    {
        "job_search": "cv_search",
        "direct": "direct"
    }
)


graph.add_edge(
    "cv_search",
    "job_query"
)

graph.add_edge(
    "job_query",
    "job_search"
)

graph.add_edge(
    "job_search",
    "final_answer"
)

graph.add_edge(
    "final_answer",
    END
)

graph.add_edge(
    "direct",
    END
)

app = graph.compile()

print(
    "Agentic CV Job Search Assistant Backend Ready!"
)