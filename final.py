import streamlit as st
import os
import pdfplumber
import time
import uuid

import gspread
from google.oauth2.service_account import Credentials
import tiktoken # New import for token counting

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# New tool imports
from langchain_tavily import TavilySearch
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain_community.tools.wolfram_alpha import WolframAlphaQueryRun

# -----------------------------
# Initialize API keys from secrets
# -----------------------------
try:
    groq_api_key = st.secrets["GROQ"]["API_KEY"]
    tavily_api_key = st.secrets["TAVILY"]["API_KEY"]
    wolfram_alpha_app_id = st.secrets["WOLFRAM"]["APP_ID"]
    os.environ["TAVILY_API_KEY"] = tavily_api_key
except KeyError:
    st.error("API keys not found in .streamlit/secrets.toml. Please add them correctly.")
    st.stop()

# -----------------------------
# Google Sheets Client Setup (Same as original)
# -----------------------------
def get_gsheet_client():
    credentials_dict = st.secrets["gcp_service_account"]
    sheet_id = credentials_dict["sheet_id"]

    credentials = Credentials.from_service_account_info(
        credentials_dict,
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    client = gspread.authorize(credentials)
    return client, sheet_id

def log_performance(query_id, mode, query, response_text, duration, status):
    try:
        client, sheet_id = get_gsheet_client()
        sheet = client.open_by_key(sheet_id).sheet1

        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
        row = [
            query_id,
            timestamp,
            mode,
            query,
            len(response_text),
            int(duration * 1000),
            status
        ]
        sheet.append_row(row)
    except Exception as e:
        st.error(f"Error logging data to Google Sheets: {e}")

# -----------------------------
# Define LLMs
# -----------------------------
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")
# Get the tokenizer for the chosen model to estimate token count
tokenizer = tiktoken.get_encoding("cl100k_base") 
# Llama-3.1-8b-instant has a 128k context window, but to be safe and account for
# prompt overhead and generation, we'll use a smaller limit for our input.
MAX_INPUT_TOKENS = 6000 # The requested limit mentioned in your error

def get_token_count(text):
    return len(tokenizer.encode(text))

# -----------------------------
# PDF Processing Functions (Same as original)
# -----------------------------
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def trim_text(text, max_tokens=3000):
    tokens = text.split()
    if len(tokens) > max_tokens:
        return ' '.join(tokens[:max_tokens])
    return text

# **Modified** to return only the retriever for better control
def process_pdf_with_langchain(pdf_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.create_documents([pdf_text])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return retriever

# -----------------------------
# Research & Math Agents (Updated)
# -----------------------------
tavily_search = TavilySearch(max_results=10)

arxiv_wrapper = ArxivAPIWrapper(
    top_k_results=5,
    doc_content_chars_max=50000
)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wolfram_alpha_tool = WolframAlphaQueryRun(
    appid=wolfram_alpha_app_id
)

tools = [
    Tool(
        name="Tavily Search",
        description="A search tool to get the latest information from the web.",
        func=tavily_search.invoke
    ),
    Tool(
        name="Arxiv Search",
        description="A search tool for academic papers on Arxiv. Use this for questions about science, physics, mathematics, or computer science.",
        func=arxiv_tool.run
    ),
    Tool(
        name="Wolfram Alpha",
        description="A tool for complex mathematical calculations, scientific data, and factual information. Use this for math problems or scientific queries.",
        func=wolfram_alpha_tool.run
    )
]

research_agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("BrainyBot: Your Intelligent Research Assistant")
st.markdown("This app can answer questions from a PDF, perform advanced research, and solve complex math and science problems.")

pdf_file = st.file_uploader("Upload a PDF for Q&A:", type=["pdf"])

user_query = st.text_input("Enter your query:", "")

if user_query:
    query_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    if pdf_file and st.session_state.get('pdf_processed_id') == pdf_file.file_id:
        # Handle PDF Q&A mode if a PDF is uploaded
        with st.spinner("Processing PDF and generating response..."):
            try:
                # Retrieve relevant documents from the vector store
                retriever = st.session_state.get('retriever')
                docs = retriever.get_relevant_documents(user_query)
                
                # Check for token count and trim documents if necessary
                prompt_template = """Use the following pieces of context to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
                
                Context: {context}
                
                Question: {question}
                
                Helpful Answer:"""
                
                context_str = "\n\n".join([doc.page_content for doc in docs])
                total_tokens = get_token_count(prompt_template.format(context=context_str, question=user_query))

                # Truncate context if it exceeds the limit
                while total_tokens > MAX_INPUT_TOKENS and len(docs) > 0:
                    docs.pop() # Remove the last document
                    context_str = "\n\n".join([doc.page_content for doc in docs])
                    total_tokens = get_token_count(prompt_template.format(context=context_str, question=user_query))
                
                if not docs:
                    response = "I'm sorry, I could not find enough relevant information in the document to answer your question."
                else:
                    final_prompt = prompt_template.format(context=context_str, question=user_query)
                    response = llm.invoke(final_prompt).content
                
                st.write("Chatbot:", response)
                duration = time.perf_counter() - start_time
                log_performance(query_id, "PDF Q&A", user_query, response, duration, "Success")
            except Exception as e:
                st.error(f"An error occurred during PDF Q&A: {e}")
                duration = time.perf_counter() - start_time
                log_performance(query_id, "PDF Q&A", user_query, str(e), duration, "Error")

    else:
        # Use the unified agent for all other queries
        with st.spinner("Thinking..."):
            try:
                response = research_agent.run(user_query)
                st.write("Response:", response)
                duration = time.perf_counter() - start_time
                log_performance(query_id, "Unified Agent", user_query, response, duration, "Success")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                duration = time.perf_counter() - start_time
                log_performance(query_id, "Unified Agent", user_query, str(e), duration, "Error")

if st.button("Clear App"):
    st.session_state.clear()
    st.experimental_rerun()

# This is an optimized block to process the PDF only once per upload
if pdf_file and st.session_state.get('pdf_processed_id') != pdf_file.file_id:
    st.session_state['pdf_processed_id'] = pdf_file.file_id
    pdf_path = "uploaded_pdf.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.read())
    
    pdf_text = extract_text_from_pdf(pdf_path)
    trimmed_pdf_text = trim_text(pdf_text)
    
    if trimmed_pdf_text:
        retriever = process_pdf_with_langchain(trimmed_pdf_text)
        st.session_state['retriever'] = retriever # Store the retriever, not the full chain
        st.success("PDF processed successfully! You can now ask questions about the document.")
    else:
        st.error("Could not extract text from the PDF. Please try a different file.")
    
    if os.path.exists(pdf_path):
        os.remove(pdf_path)