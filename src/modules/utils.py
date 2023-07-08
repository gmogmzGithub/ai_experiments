import os
import streamlit as st
import pdfplumber
from modules.chatbot import Chatbot
from modules.embedder import Embedder
from dotenv import find_dotenv, load_dotenv
import logging

load_dotenv(find_dotenv())

logging_level = os.getenv('LOGGING_LEVEL')
# Convert the string logging level to an integer constant
logging_level = logging.getLevelName(logging_level)

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)


class Utilities:
    @staticmethod
    def load_api_key():
        """
        Loads the OpenAI API key from the .env file or 
        from the user's input and returns it
        """
        api_key = st.session_state.get("api_key")

        if os.path.exists(".env") and os.environ.get("OPENAI_API_KEY") is not None:
            user_api_key = os.environ["OPENAI_API_KEY"]
            st.sidebar.success("API key loaded from .env", icon="🚀")
        elif api_key is not None:
            user_api_key = api_key
            st.sidebar.success("API key loaded from previous input", icon="🚀")
        else:
            user_api_key = st.sidebar.text_input(
                label="#### Your OpenAI API key 👇", placeholder="sk-...", type="password"
            )
            if user_api_key:
                st.session_state.api_key = user_api_key

        return user_api_key

    @staticmethod
    def handle_upload(file_types):
        """
        Handles and display uploaded_file
        :param file_types: List of accepted file types, e.g., ["csv", "pdf", "txt"]
        """
        uploaded_file = st.sidebar.file_uploader("upload", type=file_types, label_visibility="collapsed")
        if uploaded_file is not None:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()

            if file_extension == ".pdf":
                Utilities.show_pdf_file(uploaded_file)
            elif file_extension == ".txt":
                Utilities.show_txt_file(uploaded_file)
        else:
            st.session_state["reset_chat"] = True

        return uploaded_file

    @staticmethod
    def show_pdf_file(uploaded_file):
        file_container = st.expander("Your PDF file :")
        with pdfplumber.open(uploaded_file) as pdf:
            pdf_text = ""
            for page in pdf.pages:
                pdf_text += page.extract_text() + "\n\n"
        file_container.write(pdf_text)

    @staticmethod
    def show_txt_file(uploaded_file):
        file_container = st.expander("Your TXT file:")
        uploaded_file.seek(0)
        content = uploaded_file.read().decode("utf-8")
        file_container.write(content)

    @staticmethod
    def setup_chatbot_with_faiss(model, temperature):
        """
        Sets up the chatbot with the uploaded file, model, and temperature
        """
        with st.spinner("Processing..."):
            embeds = Embedder()
            vectors = embeds.faiss_contexts()

            # Create a Chatbot instance with the specified model and temperature
            chatbot = Chatbot(model, temperature, vectors)
        st.session_state["ready"] = True

        return chatbot

    @staticmethod
    def setup_chatbot_with_file(uploaded_file, model, temperature):
        """
        Sets up the chatbot with the uploaded file, model, and temperature
        """
        embeds = Embedder()

        with st.spinner("Processing..."):
            uploaded_file.seek(0)
            file = uploaded_file.read()
            # Get the document embeddings for the uploaded file
            vectors = embeds.getDocEmbeds(file, uploaded_file.name)

            # Create a Chatbot instance with the specified model and temperature
            chatbot = Chatbot(model, temperature, vectors)
        st.session_state["ready"] = True

        return chatbot
