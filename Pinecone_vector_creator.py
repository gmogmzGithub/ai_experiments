import os
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.vectorstores import Pinecone
import logging, pickle
from tqdm import tqdm
from pypdf.errors import DependencyError, PdfStreamError
from dotenv import find_dotenv, load_dotenv
from pinecone.core.client.exceptions import ServiceException
import time

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
        datasets_dir = os.getenv('DATASETS')
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

    return openai_key, embeddings_dir, pinecone_api_key, pinecone_env, index_name, datasets_dir, model


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


if __name__ == "__main__":
    openai_key, embeddings_dir, pinecone_api_key, pinecone_env, index_name, datasets_dir, model = load_dotenv_variables()
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

    vectorstore = create_or_load_pinecone_index(index_name, embeddings)

    max_retries = 5
    base_wait_time = 1  # wait time in seconds

    for attempt in range(max_retries):
        try:
            # if vectorstore is None:
            logger.info(f"No existing index found. Loading PDFs and creating new index.")
            documents = load_pdf_files(datasets_dir, 112700)
            logger.info(f"No existing index found. Loading PDFs and creating new index.")
            logger.info(f"{len(documents)} documents to be loaded into Pinecone")
            docsearch = Pinecone.from_documents(documents, embeddings, index_name=index_name)
            break  # if the request was successful, we break out of the loop
        except ServiceException as e:
            logger.error(f"Pinecone service exception occurred: {e}")
            wait_time = base_wait_time * 2 ** attempt  # exponential backoff
            logger.info(f"Waiting for {wait_time} seconds before retrying...")
            time.sleep(wait_time)
    else:
        # if we've exhausted all retries and still haven't succeeded, we raise an exception
        raise Exception("Failed to create Pinecone documents after multiple retries")