import streamlit as st
from chatbot import get_response
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_social_support", embedding_function=embedding_model)
# You might not need this RetrievalQA directly if using ConversationalRetrievalChain in chatbot.py
# qa = RetrievalQA.from_chain_type(llm=Ollama(model="qwen2.5:1.5b"), retriever=db.as_retriever())


st.title("Chatbot de Apoyos Sociales para Adultos Mayores 👴👵")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "first_message" not in st.session_state:
    st.session_state.first_message = True

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initial welcome message for a new session
if st.session_state.first_message:
    welcome = "¡Hola! Estoy aquí para ayudarte con información sobre apoyos sociales para adultos mayores. ¿En qué puedo asistirte hoy?"
    with st.chat_message("assistant"):
        st.markdown(welcome)
    st.session_state.messages.append({"role": "assistant", "content": welcome})
    st.session_state.first_message = False

# Process user input
if prompt := st.chat_input("¿Cómo puedo ayudarte?"):
    # Display user message in chat history
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and display assistant response immediately
    with st.chat_message("assistant"):
        with st.spinner("🔍 Buscando información sobre apoyos sociales..."): # Show spinner while processing
            full_response = get_response(prompt)
        st.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})