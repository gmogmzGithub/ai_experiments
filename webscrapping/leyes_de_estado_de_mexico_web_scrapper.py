from bs4 import BeautifulSoup
from tqdm import tqdm
import logging
from dotenv import find_dotenv, load_dotenv
import requests
import os

load_dotenv(find_dotenv())

logging_level = os.getenv('LOGGING_LEVEL')
# specify local directory to save PDFs
datasets = os.environ["DATASETS_FROM_WEB_SCRAPPING"]

# Convert the string logging level to an integer constant
logging_level = logging.getLevelName(logging_level)

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)

# Specify local directory to save PDFs
folder_path = os.path.join(datasets, "estado_de_mexico")
os.makedirs(folder_path, exist_ok=True)

base_url = "http://www.secretariadeasuntosparlamentarios.gob.mx/mainstream/Actividad/legislacion/leyes/pdf/{}.pdf"

fail_count = 0
for i in tqdm(range(500), desc="Downloading PDFs"):  # goes from 000 to 500
    pdf_url = base_url.format(str(i).zfill(3))  # pad with zeros to 3 digits
    response = requests.get(pdf_url)

    if response.status_code != 200:
        fail_count += 1
        if fail_count >= 10:
            logger.debug("Stopping after 10 consecutive failed downloads.")
            break
        else:
            continue

    fail_count = 0

    # Specify the local path where the PDF will be saved
    path = os.path.join(folder_path, "{}.pdf".format(str(i).zfill(3)))
    try:
        # write the PDF data to a file
        with open(path, 'wb') as f:
            f.write(response.content)
    except IOError as e:
        logger.error(f'Error saving pdf_file: {e}')
