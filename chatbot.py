from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate # Import PromptTemplate
from langchain.memory import ConversationBufferMemory # To manage chat history effectively

# Cargar el índice de Chroma previamente creado
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_social_support", embedding_function=embedding_model)

# Cargar el modelo de IA (por ejemplo, Ollama)
llm = Ollama(model="qwen2.5:1.5b")

# Initialize memory for conversational chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define a custom prompt template
template = """Eres un asistente de chatbot útil, amigable y respetuoso que proporciona información ÚNICAMENTE sobre apoyos sociales para adultos mayores en México.
Responde a las preguntas basándote ÚNICAMENTE en la información proporcionada en el siguiente contexto.
Si la pregunta no se puede responder o no tiene nada que ver con los programas sociales de la información proporcionada, indica que no tienes esa información en tu base de conocimientos y no intentes inventar una respuesta.
Asegúrate de que tus respuestas sean claras, concisas y siempre en español. Cuando se hable de programas, programas sociales o apoyos sociales, asegúrate de que la información sea precisa y relevante.

Historial de Conversación:
{chat_history}

Contexto:
{context}

Pregunta: {question}
Respuesta útil:"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question", "chat_history"], template=template)


# Crear la cadena de pregunta-respuesta (QA) con el modelo de RAG
# We'll use a slightly different chain setup with memory and prompt
qa = ConversationalRetrievalChain.from_llm(
    llm,
    db.as_retriever(),
    memory=memory, # Pass the memory
    combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT} # Apply the custom prompt
)

# No need for a global chat_history list if using ConversationBufferMemory
# chat_history = [] # This line can be removed as memory handles it

def get_response(prompt):
    # The 'qa' chain now handles chat_history internally via 'memory'
    # The 'run' method will pass the question and internally manage chat_history
    res = qa.invoke({"question": prompt}) # Use invoke with dictionary for the chain
    
    # The response object from invoke will have a 'answer' key for the result
    response_text = res["answer"]

    response_text += "\n\n---\n✨ *Información obtenida de fuentes de apoyos sociales para adultos mayores.*"
    return response_text