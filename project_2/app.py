import streamlit as st

# Vector store to store and retrieve embeddings efficiently using FAISS
from langchain.vectorstores import FAISS

# Generate text embeddings using OpenAI or Hugging Face models
from langchain.embeddings import  SentenceTransformerEmbeddings

# Use local LLMs (e.g., via Ollama) for response generation
from langchain.llms import Ollama

# Build a retrieval chain that combines a retriever, a prompt, and an LLM
from langchain.chains import ConversationalRetrievalChain

# Create prompts for the RAG system
from langchain.prompts import PromptTemplate

# Load the FAISS vector store
embeddings = SentenceTransformerEmbeddings(model_name="thenlper/gte-small")
vectordb = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)  # Use the same embedding as before
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 8})

# Initialize the LLM
llm = Ollama(model="gemma3:1b", temperature=0.1)

# Define the prompt template
SYSTEM_TEMPLATE = """
You are a **Customer Support Chatbot**. Use only the information in CONTEXT to answer.
If the answer is not in CONTEXT, respond with “I'm not sure from the docs.”

Rules:
1) Use ONLY the provided <context> to answer.
2) If the answer is not in the context, say: "I don't know based on the retrieved documents."
3) Be concise and accurate. Prefer quoting key phrases from the context.
4) When possible, cite sources as [source: source] using the metadata.

CONTEXT:
{context}

USER:
{question}
"""

prompt = PromptTemplate(
    template=SYSTEM_TEMPLATE,
    input_variables=["context", "question"],
)

# Build the ConversationalRetrievalChain
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    condense_question_prompt=prompt, 
)

# Streamlit UI
st.title("Customer Support Chatbot")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", "")

if user_input:
    result = chain({
        "question": user_input,
        "chat_history": st.session_state.chat_history,
    })
    st.session_state.chat_history = result.get("chat_history", [])
    st.write("Bot:", result["answer"])