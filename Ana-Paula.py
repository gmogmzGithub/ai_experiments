import streamlit as st
import os
import pinecone
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import Pinecone
import logging
import pickle
from tqdm import tqdm
from pypdf.errors import DependencyError, PdfStreamError
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
logging_level = os.getenv('LOGGING_LEVEL') or 'INFO'

# Configure logging
logging.basicConfig(level=logging.WARN, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)


def load_dotenv_variables():
    try:
        openai_key = os.environ["OPENAI_API_KEY"]
        embeddings_dir = os.getenv("EMBEDDINGS")
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        pinecone_env = os.getenv('PINECONE_ENV')
        index_name = os.getenv('PINECONE_INDEX')
        datasets = os.getenv('DATASETS')
        model = os.getenv('CHATGPT_MODEL')
    except KeyError as e:
        logger.error(f"Missing environment variable: {str(e)}")
        raise

    # Pinecone initialization
    logger.info("Initializing Pinecone")
    try:
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
    except Exception as e:  # replace Exception with a more specific exception if possible
        logger.error("Failed to initialize Pinecone", exc_info=True)
        raise

    return openai_key, embeddings_dir, pinecone_api_key, pinecone_env, index_name, datasets, model


def load_pdf_files(datasets_dir, documents_limit):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    )

    pickle_file = datasets_dir + '/documents.pkl'
    logger.debug(pickle_file)

    if os.path.exists(pickle_file):
        logger.info(f"Loading documents from pickle file: {pickle_file}")
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)

    documents = []
    temp_documents = []

    # Create a list of all directories in the dataset directory
    dirs = [d for d in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, d))]
    dirs = sorted(dirs)  # sort directories in alphabetical order

    # Iterate over all directories
    for dir in dirs:
        logger.debug(f"Going over the directory: '{dir}'")
        dir_path = os.path.join(datasets_dir, dir)

        # Create a list of all PDF files in the directory
        pdf_files = []
        for filename in os.listdir(dir_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(dir_path, filename)
                pdf_files.append(file_path)

        # Then iterate over all PDF files with tqdm progress bar
        for file_path in tqdm(pdf_files, desc="Loading PDF files", unit="file"):
            if len(temp_documents) >= documents_limit:  # stop when documents limit is reached
                logger.debug(f"Breaking out of the loop while on {dir}")
                break
            loader = UnstructuredPDFLoader(file_path)
            try:
                document_pages = loader.load()
                documents.extend(document_pages)
                temp_documents.extend(document_pages)
                temp_documents = text_splitter.split_documents(temp_documents)
                percentage = (len(temp_documents) * 100)/documents_limit
                logger.debug(f"So far... we have loaded {len(temp_documents)} documents out of {documents_limit} capacity of {percentage}%")
            except PdfStreamError:
                logger.error(f"Corrupted or invalid PDF: {file_path}")
                os.remove(file_path)  # Delete the file if it's invalid or corrupted
                continue
            except DependencyError:
                logger.error(
                    "PyCryptodome is required for decrypting some PDF files. "
                    "Please install it by running: pip install pycryptodome")
                return []  # Or handle this situation appropriately

        if len(temp_documents) >= documents_limit:  # stop when documents limit is reached
            break

    logger.debug(f"{len(documents)} documents loaded")

    documents = text_splitter.split_documents(documents)

    # Save documents to a pickle file
    logger.info(f"Saving documents to pickle file: {pickle_file}")
    with open(pickle_file, 'wb') as f:
        pickle.dump(documents, f)

    return documents


def create_or_load_pinecone_index(index_name, embeddings):
    try:
        if index_name in pinecone.list_indexes():
            logger.info("Loading existent vectors from Pinecone")
            return Pinecone.from_existing_index(index_name, embeddings)
        else:
            logger.info("Creating vectors...")
            logger.debug("Vectors do not exist, we will create them")
            pinecone.create_index(index_name, dimension=1536, metric="euclidean")
    except pinecone.core.client.exceptions.ForbiddenException as e:
        logger.error(f"Failed to interact with Pinecone due to authorization error: {e}", exc_info=True)
        raise
    except Exception as e:  # catch-all for other exceptions
        logger.error(f"Failed to interact with Pinecone: {e}", exc_info=True)
        raise
    return None


def default_prompt():
    return """Hola, soy la Licenciada en Derecho Ana Paula, tu asistente legal virtual. Estoy aquí para ayudarte a
    entender documentos legales basados en la ley mexicana. ¿Cómo puedo asistirte hoy? """


def initialize_chat_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = [default_prompt()]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]


def conversational_chat(query):
    try:
        result = chain({"question": query, "chat_history": st.session_state['history']})
    except Exception as e:  # replace Exception with a more specific exception if possible
        logger.error("Failed to generate chatbot response", exc_info=True)
        return "I'm sorry, I couldn't process your request. Please try again."

    logger.info(f"Query: {query}")
    logger.info(f"Answer: {result['answer']}")

    st.session_state['history'].append((query, result["answer"]))

    return result["answer"]


if __name__ == "__main__":
    openai_key, embeddings_dir, pinecone_api_key, pinecone_env, index_name, datasets_dir, model = load_dotenv_variables()
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

    vectorstore = create_or_load_pinecone_index(index_name, embeddings)

    if vectorstore is None:
        logger.info(f"No existing index found. Loading PDFs and creating new index.")
        documents = load_pdf_files(datasets_dir, 112700)
        logger.info(f"No existing index found. Loading PDFs and creating new index.")
        logger.info(f"{len(documents)} documents to be loaded into Pinecone")
        vectorstore = Pinecone.from_documents(documents, embeddings, index_name=index_name)
        logger.info('Pinecone index created')

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4})

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

    llm = ChatOpenAI(model_name=model, temperature=0.0)
    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context", "question"])
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        verbose=True,
        return_source_documents=True,
        max_tokens_limit=4097,
        combine_docs_chain_kwargs={'prompt': QA_PROMPT}
    )

    initialize_chat_state()

    # container for the chat history
    response_container = st.container()
    # container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Conversemos sobre los documentos", key='input')
            submit_button = st.form_submit_button(label='Send')

            if submit_button:
                if not user_input:
                    st.error("Please enter a valid query.")
                else:
                    output = conversational_chat(user_input)
                    st.session_state['past'].append(user_input)
                    st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
