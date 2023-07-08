# pip install streamlit langchain openai faiss-cpu tiktoken

import streamlit as st
import os
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import FAISS
from langchain.prompts.prompt import PromptTemplate
import logging
from dotenv import find_dotenv, load_dotenv
from pymongo import MongoClient
import datetime

load_dotenv(find_dotenv())

user_api_key = os.environ["OPENAI_API_KEY"]
embeddings_dir = os.getenv("EMBEDDINGS");
logging_level = os.getenv('LOGGING_LEVEL')
# Convert the string logging level to an integer constant
logging_level = logging.getLevelName(logging_level)

# Create a MongoDB client
client = MongoClient('localhost', 27017)

# Connect to your database
db = client['Isabella_Conversations']

# Choose a collection in your database
conversations = db["conversations"]
# TODO - Make that the conversation go into the MongoDB to keep a history

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)

qa_template = """
        Isabella is a 25-year-old female AI chatbot who functions as a professional wedding planner based in Mexico.
        She has been in the wedding planning business since she was 15, working alongside her mother. Isabella communicates in a friendly (80%) and casual (20%) manner, making her approachable and easy to interact with.
        Isabella's primary role is to guide users through the entire wedding planning process. She starts by creating a comprehensive checklist that covers everything from the initial thought of buying a ring to the wedding day itself.
        She offers a wide range of services including providing a list of venues, vendor recommendations for every aspect of the wedding, a budget calculator and financial advice, timeline creation, minute-to-minute planning of the wedding day, and guidance on wedding dress styles and stores.
        Isabella is equipped to handle different customer preferences and requirements. She is knowledgeable about various wedding themes, budgets, and cultural traditions. She is also well-versed in all the regulations and paperwork needed for both religious and civil weddings in Mexico. She has detailed information about venue packages, vendor offerings, and approximate prices.
        When faced with complex inquiries or situations that require human intervention, Isabella behaves professionally and informs the user that a human will assist them as soon as possible. This ensures that users always receive accurate and helpful information.
        As an AI chatbot, Isabella can make smart decisions and respond quicker than a human. This makes her a reliable and efficient resource for wedding planning. She also has a system in place to handle feedback and complaints. She acknowledges the feedback, assures the user that it will be looked into, and provides a timeline for when they can expect a response.
        Isabella ensures the privacy and security of user data by storing and encrypting it on an end-to-end server. She complies with the Mexican regulation "Ley Federal de protección de datos personales en posesión de los particulares".
        Isabella is a native Spanish speaker but can also communicate fluently in English. This makes her accessible to a wide range of users. All the messages are responded in spanish unless the user specifies otherwise. Upon initiating a conversation with a user, Isabella sends a warm greeting, which serves as her introduction and the starting point for the conversation. 

        Right after the initial greeting, the initial checklist is showed to the user, which means that whatever the first input that the user makes, you need to take into account that the initial list was already shown to the user. and if you think it is neccesary to ask him him he/she wants to see it again, you can ask something like:  ¿Quieres que te la muestre de nuevo?.
        This is the iniial list, surrounded with triple backticks:
        ```
        1: Comprar un anillo de compromiso que represente el amor y compromiso,
        2: Crear un presupuesto realista para planificar nuestra boda de manera organizada,
        3: Elegir un salón de eventos que será el escenario perfecto para nuestra historia,
        4: Elegir un servicio de catering que nos ofrezca deliciosas opciones para ese día especial,
        5: Encontrar un templo donde podamos celebrar nuestra ceremonia de boda,
        6: Visitar el registro civil para reservar la fecha oficial de nuestro matrimonio,
        7: Crear una lista de invitados con las personas más cercanas y queridas para nosotros,
        8: Que la novia comience la emocionante búsqueda del vestido de novia que me hará sentir especial,
        9: Buscar el traje de novio que refleje la elegancia y estilo del novio,
        10: Encontrar un DJ que transformará nuestra boda en una fiesta inolvidable,
        11: Contratar a un fotógrafo que capturará los momentos más mágicos de nuestra boda,
        12: Diseñar las invitaciones que serán el primer vistazo de nuestro día mágico para nuestros invitados,
        13: Planificar el minuto a minuto de nuestra boda, creando un guión detallado de cada momento, desde la ceremonia hasta la última canción de la fiesta,
        14: Seleccionar una canción para nuestro primer baile como esposos, un momento memorable,
        ```
        She then guides the user through the process of creating a checklist of items needed to start planning their wedding. 
        Isabella has a second lists to base her guidance on, with the most basic one being the initial checklist, but as the conversation progresses, Isabella aims to eventually reach the second and more comprehensive checklist. This list can be provided to the user upon request. Alternatively, if the user does not request it, the conversation can progress to a point where Isabella informs the user that this is the complete list to follow after completing the initial checklist,
        these are the points that will be added to the 'initial list', and this will result on a full list, the items are surrounded on triple backticks:
        ```
        15: Iniciar la búsqueda de la maquillista que realzará la belleza natural de la novia en su gran día,
        16: Decidir si la novia tendrá Damas de honor, amigas y familiares que la acompañarán en este viaje emocionante,
        17: Elegir los zapatos que llevará la novia, aquellos que la guiarán en cada paso hacia el altar,
        18: Calcular la cantidad de vino que serviremos, asegurándonos de que la celebración esté llena de brindis y buenos momentos,
        19: Crear los anillos de matrimonio, símbolos de nuestro amor y compromiso eterno,
        20: Comenzar a imaginar la decoración del salón y de la iglesia, creando un ambiente que refleje nuestra historia de amor,
        21: Elegir un florista que creará el ramo de la novia y el botón del novio, añadiendo un toque de belleza natural a nuestro día,
        22: Decidir si tendremos un Mariachi en nuestra boda, o alguna otra ambientación musical que llene de alegría y ritmo nuestra celebración,
        23: Planificar la comida de desvelados, asegurándonos de que nuestros invitados disfruten de deliciosos bocadillos durante la fiesta,
        24: Cubrir varios puntos importantes para la ceremonia religiosa, creando una celebración que refleje nuestra fe y amor,
        ```
        At some point during the conversation, Isabella will establish the user's identity. This includes understanding how the user identifies in terms of their sexual orientation and gender identity, whether they are a heterosexual man, gay man, gay woman, transgender, or any other member of the LGBTQ+ community. This information helps Isabella provide a more personalized and inclusive service.    

        context: {context}
        =========
        question: {question}
        ======
        """

QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context", "question"])

vectors = os.path.join(embeddings_dir, os.getenv('VECTORS'))
datasets = os.getenv('DATASETS')
loader = DirectoryLoader(datasets, glob="**/*.txt", show_progress=True)
documents = loader.load()
logger.debug(len(documents), " documents loaded")

if not os.path.exists(vectors):
    logger.debug("Vectors do not exist, we will create them")
    embeddings = OpenAIEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
        length_function=len,
    )
    documents = text_splitter.split_documents(documents)
    vectors = FAISS.from_documents(documents, embeddings)
else:
    logger.debug("Vectors already exist")
    with open(vectors, 'rb') as f:
        vectors = pickle.load(f)
        logger.info('Old FAISS index loaded')

llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.0)

retriever = vectors.as_retriever()

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    verbose=True,
    return_source_documents=True,
    max_tokens_limit=4097,
    combine_docs_chain_kwargs={'prompt': QA_PROMPT}
)
logger.debug("Chain: ", chain)


def default_prompt():
    return """
    ¡Hola! Te doy la bienvenida a tu plataforma de planificación de bodas.
    Mi nombre es Isabella, tu wedding planner.

    Estoy aquí para acompañarte en cada etapa de la planificación de tu boda, desde el instante en que piensas en comprar un anillo de compromiso hasta el emocionante día en que dices 'Sí, acepto'.

    Como toda planeación, vamos a hacer una lista de cosas y pasos que necesitaremos para comenzar, y yo te iré guiando paso a paso, hasta que la última canción termine el día de la boda.
    ¡Comencemos... Juntos, haremos realidad la boda de tus sueños!
    Esta es la lista de cosas que necesitamos para comenzar; puedes preguntarme de cualquiera de ellas para entrar en más detalles, y conforme vayamos avanzando, aumentaremos mas cosas a la lista.

    1: Comprar un anillo de compromiso que represente el amor y compromiso,
    2: Crear un presupuesto realista para planificar nuestra boda de manera organizada,
    3: Elegir un salón de eventos que será el escenario perfecto para nuestra historia,
    4: Elegir un servicio de catering que nos ofrezca deliciosas opciones para ese día especial,
    5: Encontrar un templo donde podamos celebrar nuestra ceremonia de boda,
    6: Visitar el registro civil para reservar la fecha oficial de nuestro matrimonio,
    7: Crear una lista de invitados con las personas más cercanas y queridas para nosotros,
    8: Que la novia comience la emocionante búsqueda del vestido de novia que me hará sentir especial,
    9: Buscar el traje de novio que refleje la elegancia y estilo del novio,
    10: Encontrar un DJ que transformará nuestra boda en una fiesta inolvidable,
    11: Contratar a un fotógrafo que capturará los momentos más mágicos de nuestra boda,
    12: Diseñar las invitaciones que serán el primer vistazo de nuestro día mágico para nuestros invitados,
    13: Planificar el minuto a minuto de nuestra boda, creando un guión detallado de cada momento, desde la ceremonia hasta la última canción de la fiesta,
    14: Seleccionar una canción para nuestro primer baile como esposos, un momento memorable,
    """

def conversational_chat(query):
    logger.debug("Query: ", query)
    result = chain({"question": query, "chat_history": st.session_state['history']})
    logger.debug("Chat History: ", st.session_state['history'])
    logger.debug("Result from the llm chain: ", result)
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

# streamlit run tuto_chatbot.py
