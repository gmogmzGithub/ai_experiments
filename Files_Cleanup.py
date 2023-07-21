import os
from langchain.document_loaders import PyPDFLoader
import logging
from pypdf.errors import PdfStreamError
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm

load_dotenv(find_dotenv())

logging_level = os.getenv('LOGGING_LEVEL')
# Configure logging
logging_level = logging.getLevelName(logging_level)
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)

if __name__ == "__main__":
    datasets = os.getenv('DATASETS')

    # first collect all pdf files
    pdf_files = []
    for dirpath, dirnames, filenames in os.walk(datasets):
        for filename in filenames:
            if filename.endswith(".pdf"):
                pdf_files.append(os.path.join(dirpath, filename))

    # then process them
    for file_path in tqdm(pdf_files, desc="Processing PDFs"):
        loader = PyPDFLoader(file_path=file_path)
        try:
            # Attempt to load the PDF
            loader.load()
        except PdfStreamError:
            # If loading fails, delete the file
            logger.error(f"Corrupted or invalid PDF: {file_path}, deleting...")
            os.remove(file_path)
