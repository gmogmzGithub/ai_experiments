import os
import tempfile
import pickle
import hashlib

from langchain.document_loaders import DirectoryLoader

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import find_dotenv, load_dotenv
import logging

load_dotenv(find_dotenv())
embeddings_dir = os.getenv("EMBEDDINGS");
logging_level = os.getenv('LOGGING_LEVEL')
# Convert the string logging level to an integer constant
logging_level = logging.getLevelName(logging_level)

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)


class Embedder:

    def __init__(self):
        self.PATH = "embeddings"
        self.createEmbeddingsDir()

    def createEmbeddingsDir(self):
        """
        Creates a directory to store the embeddings vectors
        """
        if not os.path.exists(self.PATH):
            os.mkdir(self.PATH)

    def storeDocEmbeds(self, file, original_filename):
        """
        Stores document embeddings using Langchain and FAISS
        """
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
            tmp_file.write(file)
            tmp_file_path = tmp_file.name

        def get_file_extension(uploaded_file):
            file_extension = os.path.splitext(uploaded_file)[1].lower()

            return file_extension

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100,
            length_function=len,
        )

        file_extension = get_file_extension(original_filename)

        if file_extension == ".csv":
            loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
                'delimiter': ',', })
            data = loader.load()

        elif file_extension == ".pdf":
            loader = PyPDFLoader(file_path=tmp_file_path)
            data = loader.load_and_split(text_splitter)

        elif file_extension == ".txt":
            loader = TextLoader(file_path=tmp_file_path, encoding="utf-8")
            data = loader.load_and_split(text_splitter)

        embeddings = OpenAIEmbeddings()

        vectors = FAISS.from_documents(data, embeddings)
        os.remove(tmp_file_path)

        # Save the vectors to a pickle file
        with open(f"{self.PATH}/{original_filename}.pkl", "wb") as f:
            pickle.dump(vectors, f)

    def faiss_contexts(self):
        documents_hash = os.getenv('DOCUMENTS_HASH_FILE')
        datasets = os.getenv('DATASETS')
        loader = DirectoryLoader(datasets, glob="**/*.txt", show_progress=True)
        documents = loader.load()
        logger.debug('Documents loaded with size: ', len(documents))
        documents_bytes = pickle.dumps(documents)
        hash_object = hashlib.sha256(documents_bytes)
        hex_dig = hash_object.hexdigest()

        # current_documents_hash = None
        # if os.path.exists(documents_hash):
        #     logger.debug('The hash of the documents does exist')
        #     with open(os.path.join(embeddings_dir, documents_hash), "r") as f:
        #         current_documents_hash = f.read().strip()

        vectors = os.path.join(embeddings_dir, os.getenv('VECTORS'))

        if logging_level == 10: # DEBUG
            if not os.path.exists(vectors):
                logger.info("⚠️ Vectors does not exists ⚠")
            else:
                logger.info("✅ Vectors DO exists 🫡")
            # if current_documents_hash is None:
            #     logger.info("⚠️ The HASH of the documents does not exists ⚠")
            # else:
            #     logger.info("✅ The HASH of the documents DO exists 🫡")
            # if current_documents_hash != hex_dig:
            #     logger.info("⚠️ The current HASH of the documents differ from the previous one; "
            #                 "something has changed on the datasets ⚠")
            # else:
            #     logger.info("✅ The current HASH of the documents IS the SAME from the previous one 🫡")

        # if current_documents_hash is None or current_documents_hash != hex_dig or not os.path.exists(vectors):
        if not os.path.exists(vectors):
            logger.info('Vectors DB will be created for the documents since something changed')
            embeddings = OpenAIEmbeddings()

            with open(os.path.join(embeddings_dir, documents_hash), "w") as f:
                f.write(hex_dig)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=100,
                length_function=len,
            )

            documents = text_splitter.split_documents(documents)
            db = FAISS.from_documents(documents, embeddings)

            # Save FAISS index
            with open(vectors, 'wb') as f:
                pickle.dump(db, f)
                logger.info('FAISS index created')
            return db
        else:
            # If the hash hasn't changed, load the 'db' object from the 'faiss_index.pkl' file
            with open(vectors, 'rb') as f:
                db = pickle.load(f)
                logger.info('Old FAISS index loaded')
            return db

    def getDocEmbeds(self, file, original_filename):
        """
        Retrieves document embeddings
        """
        if not os.path.isfile(f"{self.PATH}/{original_filename}.pkl"):
            self.storeDocEmbeds(file, original_filename)

        # Load the vectors from the pickle file
        with open(f"{self.PATH}/{original_filename}.pkl", "rb") as f:
            vectors = pickle.load(f)

        return vectors
