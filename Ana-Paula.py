import streamlit as st
import os
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.prompts.prompt import PromptTemplate
import logging
from pypdf.errors import PdfStreamError
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

user_api_key = os.environ["OPENAI_API_KEY"]
embeddings_dir = os.getenv("EMBEDDINGS")
logging_level = os.getenv('LOGGING_LEVEL')

# Convert the string logging level to an integer constant
logging_level = logging.getLevelName(logging_level)

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)

vectors_db = os.path.join(embeddings_dir, os.getenv('VECTORS'))
datasets = os.getenv('DATASETS')

# Load PDF files
documents = []
# Iterate over the files in the directory
for dirpath, dirnames, filenames in os.walk(datasets):
    for filename in filenames:
        # Check if the file is a PDF
        if filename.endswith(".pdf"):
            # Create a full file path
            file_path = os.path.join(dirpath, filename)
            # Create a PyPDFLoader for the file
            loader = PyPDFLoader(file_path=file_path)
            try:
                # Load the document and add it to the list
                document_pages = loader.load()
                # Flatten the list of lists
                documents.extend(document_pages)
            except PdfStreamError:
                logger.error(f"Corrupted or invalid PDF: {file_path}")
                continue

logger.debug(f"{len(documents)} documents loaded")

if not os.path.exists(embeddings_dir):
    os.makedirs(embeddings_dir)

if not os.path.exists(vectors_db):
    with st.spinner('Creating vectors...'):
        logger.info("Creating vectors...")
        logger.debug("Vectors do not exist, we will create them")
        embeddings = OpenAIEmbeddings()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100,
            length_function=len,
        )
        documents = text_splitter.split_documents(documents)
        vectors = FAISS.from_documents(documents, embeddings)

        # Save FAISS index
        with open(vectors_db, 'wb') as f:
            pickle.dump(vectors, f)
        logger.info('FAISS index created')

else:
    logger.debug("Vectors already exist")
    with open(vectors_db, 'rb') as f:
        vectors = pickle.load(f)
        logger.info('Old FAISS index loaded')

llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.0)

retriever = vectors.as_retriever()

qa_template = """You are an AI chatbot designed to function as a professional paralegal, your name is Ana Paula 
specializing in various areas of Mexican law. You are equipped with a comprehensive vector database of legal documents 
specific to Mexican law, which serves as your primary reference for answering inquiries. You are capable of handling 
complex legal language and concepts, but your main task is to simplify these concepts so that anyone, even those 
without legal knowledge, can understand. 

When faced with complex queries that you cannot handle, you respond professionally, informing the user that a human 
legal expert will be contacted to address their query. Your tone is primarily formal (70%), with elements of empathy 
(20%) and friendliness (10%). 

You adhere to the Mexican law 'LEY FEDERAL DE PROTECCIÓN DE DATOS PERSONALES EN POSESIÓN DE LOS PARTICULARES', 
ensuring that all data shared in the conversation is protected. You do not keep any record of the data, 
and every conversation is destroyed when the conversation is closed. 

While you can provide information based on legal documents and Mexican law, you always clarify that you cannot give 
legal advice as you are not a licensed attorney. Your services are available to anyone, from ranch owners and college 
students to experienced lawyers, and you strive to provide clear and understandable explanations of legal documents. 

You are a bilingual Mexican AI, fluent in both Spanish and English. While you can communicate in other languages to 
accommodate expats or immigrants living in Mexico, all your responses are based on Mexican law. 

In case of errors or misunderstandings, you acknowledge the limitations of AI and assure the user that you are 
continually learning and improving. 

context: {context}
=========
question: {question}
======
"""

QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context", "question"])
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    verbose=True,
    return_source_documents=True,
    max_tokens_limit=4097,
    combine_docs_chain_kwargs={'prompt': QA_PROMPT}
)


def default_prompt():
    return """Hola, soy la Licenciada en Derecho Ana Paula, tu asistente legal virtual. Estoy aquí para ayudarte a 
    entender documentos legales basados en la ley mexicana. ¿Cómo puedo asistirte hoy? """


def conversational_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))

    return result["answer"]


if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = [default_prompt()]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]

# container for the chat history
response_container = st.container()
# container for the user's text input
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Conversemos sobre los documentos", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = conversational_chat(user_input)

        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

# streamlit run Ana-Paula.py
