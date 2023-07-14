from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import unquote
import requests
import os
import logging
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

logging_level = os.getenv('LOGGING_LEVEL')
# specify local directory to save PDFs
datasets = os.environ["DATASETS_FROM_WEB_SCRAPPING"]
leyesBiblio_url = os.environ["LEYES_CAMPECHE"]

# Convert the string logging level to an integer constant
logging_level = logging.getLevelName(logging_level)

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)

response = requests.get(leyesBiblio_url)
soup = BeautifulSoup(response.text, 'html.parser')

# Look for all <a> tags with href starting with "pdf/"
pdf_links = soup.find_all('a', href=lambda href: href and "/index.php/" in href)
pdf_hrefs = [link['href'] for link in pdf_links]

# Filter out URLs ending with .doc or .docx
pdf_hrefs = [pdf for pdf in pdf_hrefs if not (pdf.endswith('.doc')
                                              or pdf.endswith('.docx')
                                              or pdf.endswith('.zip')
                                              or '(anterior)' in pdf)]

# Modify hrefs to prepend the desired string and remove leading ".."
pdf_hrefs = ['http://legislacion.congresocam.gob.mx' + href + '/file' for href in pdf_hrefs]

pdf_file = 1
for pdf_url in tqdm(pdf_hrefs, desc="Descargando los PDFs de Campeche", unit="file"):
    try:
        # make the request to download the PDF
        response = requests.get(pdf_url)

        # raise an exception if the response status is not OK
        response.raise_for_status()
    except (requests.HTTPError, requests.ConnectionError) as e:
        logger.error(f'Error downloading {pdf_url}: {e}')
        continue

    # specify the local (absolute) path where the PDF will be saved
    folder_path = os.path.join(datasets, "campeche")
    os.makedirs(folder_path, exist_ok=True)  # create directory if it doesn't exist

    path = os.path.join(folder_path, str(pdf_file) + ".pdf")
    try:
        # write the PDF data to a file
        with open(path, 'wb') as f:
            f.write(response.content)

        # Open the file in binary mode and check if it is a valid PDF
        with open(path, 'rb') as f:
            magic_number = f.read(4)
            if magic_number != b'%PDF':
                logger.error(f'Invalid PDF {pdf_file}')
                os.remove(path)
                continue

        pdf_file = pdf_file + 1
    except IOError as e:
        logger.error(f'Error saving {pdf_file}: {e}')
