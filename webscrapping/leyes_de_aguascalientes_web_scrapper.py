from bs4 import BeautifulSoup
from tqdm import tqdm
import requests
import os
import logging
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

logging_level = os.getenv('LOGGING_LEVEL')
# specify local directory to save PDFs
datasets = os.environ["DATASETS_FROM_WEB_SCRAPPING"]
LeyesBiblio_url = os.environ["LEYES_AGUASCALIENTES"]

# Convert the string logging level to an integer constant
logging_level = logging.getLevelName(logging_level)

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)

response = requests.get(LeyesBiblio_url)
soup = BeautifulSoup(response.text, 'html.parser')

# Look for all <a> tags with href starting with "pdf/"
pdf_links = soup.find_all('a', href=lambda href: href and "agenda_legislativa/leyes/descargarPdf" in href)

# prepended string
base_url = "https://congresoags.gob.mx/"

# list to store the full URLs
full_urls = []

# append the href attribute of each link, prepended with base_url, to the list
for link in pdf_links:
    full_urls.append(base_url + link['href'])

for pdf_url in tqdm(full_urls, desc="Descargando los PDFs de Aguascalientes 🌊🥵", unit="file"):
    # get the PDF file name by splitting the URL at the slash and getting the last part
    pdf_file = pdf_url.split("/")[-1]

    try:
        # make the request to download the PDF
        response = requests.get(pdf_url)

        # raise an exception if the response status is not OK
        response.raise_for_status()
    except (requests.HTTPError, requests.ConnectionError) as e:
        logger.error(f'Error downloading {pdf_url}: {e}')
        continue

    # specify the local (absolute) path where the PDF will be saved
    folder_path = os.path.join(datasets, "aguascalientes")
    os.makedirs(folder_path, exist_ok=True)  # create directory if it doesn't exist

    path = os.path.join(folder_path, pdf_file)
    try:
        # Ensure the path ends with '.pdf'
        if not path.lower().endswith('.pdf'):
            path += '.pdf'

        # write the PDF data to a file
        with open(path, 'wb') as f:
            f.write(response.content)
    except IOError as e:
        logger.error(f'Error saving {pdf_file}: {e}')