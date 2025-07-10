import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain_groq import ChatGroq
import tempfile
import os

os.environ["GROQ_API_KEY"] = st.secrets['GROQ_API_KEY']
os.environ['HF_TOKEN']=st.secrets['HF_TOKEN']

st.title("📄 PDF Q&A Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
query = st.chat_input("Ask a question about the PDF")

if st.button("🔄 Clear Chat"):
    if "messages" in st.session_state:
        del st.session_state["messages"]
    if "store" in st.session_state:
        for history in st.session_state.store.values():
            if isinstance(history, ChatMessageHistory):
                history.clear() 
        del st.session_state["store"]
    

@st.cache_resource
def get_text_splitter():
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_llm():
    return ChatGroq(model="Llama3-8b-8192")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        st.session_state.temp_path = tmp_file.name

    # Use PyMuPDFLoader with actual file path
if "messages" not in st.session_state:
    st.session_state.messages = []
if "store" not in st.session_state:
    st.session_state.store = {}

for role,inp in st.session_state.messages:
    st.chat_message(role).write(inp)

def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id]=ChatMessageHistory()
    return st.session_state.store[session_id]
if query and "temp_path" in st.session_state:
    st.session_state.messages.append(("user",query))
    st.chat_message("user").write(query)
    loader = PyMuPDFLoader(st.session_state.temp_path)
    documents = loader.load()

   
    docs =get_text_splitter().split_documents(documents)

    
    db = FAISS.from_documents(docs, get_embeddings())

    retriever = db.as_retriever()
    llm=get_llm()
    system_prompt = (
        "You are an intelligent assistant for answering questions based on a PDF document. "
        "You must use the following retrieved context to answer. "
        "If the user's question is unclear, vague, or lacks specific reference (e.g., 'What is the use of it? ', 'What are the these','How it is'), "
        "then respond politely asking for clarification or rephrasing, instead of guessing. "
        "If the context is insufficient, say 'The information in the PDF is not sufficient.' "
        "Otherwise, provide a detailed and informative answer using the context below. "
        "{context}"
    )
    prompt_template=ChatPromptTemplate.from_messages([
        ("system",system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human","Question:{input}")
    ])
    contextualize_q_system_prompt=(
     "You are an intelligent assistant that reformulates user questions based on the conversation history.\n"
     "If the latest user question depends on previous messages, reformulate it into a standalone question using the chat history.\n"
     "If the question is already standalone and does not require context, return the given questions as-is.\n"
     "Do not answer the question. Just reformulate it. \n"
     "Keep it clear and concise.\n"
    )
    qa_prompt_template=ChatPromptTemplate.from_messages([
        ("system",contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")
    ])
    history_aware_retriever=create_history_aware_retriever(llm,retriever,qa_prompt_template)
    question_chain=create_stuff_documents_chain(llm,prompt_template)
    rag_chain=create_retrieval_chain(history_aware_retriever,question_chain)
    conversational_rag_chain=RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    response=conversational_rag_chain.invoke(
        {'input':query},
        config={
            "configurable":{"session_id":"1"}
        },)
    st.chat_message("assistant").write(response['answer'])
    st.session_state.messages.append(("assistant",response['answer']))
else:
    st.warning("Please upload a PDF before asking questions.")
