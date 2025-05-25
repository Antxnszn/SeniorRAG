# Importar las librerías necesarias
import json
from langchain_community.document_loaders import TextLoader # Updated import
from langchain_community.vectorstores import Chroma # Updated import
from langchain_community.embeddings import HuggingFaceEmbeddings # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Load the JSON data
with open('./data/data.json', 'r', encoding='utf-8') as f:
    social_support_data = json.load(f)

# Convert JSON data to a list of Document objects
documents = []
for program_name, program_details in social_support_data.items():
    content = f"Programa: {program_name}\n"
    for key, value in program_details.items():
        if value: # Only add if the value is not empty
            content += f"{key.replace('_', ' ').title()}: {value}\n"
    documents.append(Document(page_content=content, metadata={"source": program_details.get("url", "N/A"), "program_name": program_name}))

# Dividir el contenido en fragmentos más pequeños (chunks)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents)

# Crear embeddings con un modelo de Hugging Face
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Crear la base de datos Chroma con los embeddings generados
# Change the persist_directory name to reflect the new domain
db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_social_support")

# Persistir el índice para futuras consultas
db.persist()

# ¡Listo! El índice de Chroma ahora está creado y almacenado
print("¡El índice de Chroma para apoyos sociales ha sido creado con éxito!")