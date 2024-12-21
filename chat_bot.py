import os
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv
import base64

load_dotenv()  # Carrega variáveis do arquivo .env
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("A chave da API OpenAI não foi encontrada. Configure o arquivo .env.")

file_path = "TESTE.xlsx"

# Caminho para o logo
logo_image_path = "LOGO_SCAN_BRANCO.png"


# Configuração do modelo de linguagem
chat = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.8,
    openai_api_key=openai_api_key
)

# Carregar e converter o logo para Base64
try:
    with open(logo_image_path, "rb") as logo_image_file:
        logo_image_base64 = base64.b64encode(logo_image_file.read()).decode("utf-8")
except FileNotFoundError:
    st.error("O arquivo do logo não foi encontrado. Certifique-se de que 'LOGO_SCAN_BRANCO.png' está no caminho correto.")
    st.stop()

# Exibir o logo na interface
st.markdown(
    f"""
    <style>
        .logo {{
            position: absolute;
            top: -14px;
            left: -260px;
            width: 250px;
        }}
        .question {{
            background-color: #ffffff;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 5px;
        }}
        .answer {{
            background-color: #d0e7ff;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }}
    </style>
    <img src="data:image/png;base64,{logo_image_base64}" class="logo">
    """,
    unsafe_allow_html=True
)

st.title("Chatbot da Blu Logistics part of Scan Logistics")
st.write("Digite sua pergunta no campo abaixo para obter uma resposta.")

@st.cache_resource
def prepare_qdrant(data):
    loader = DataFrameLoader(data, page_content_column="pergunta")
    documents = loader.load()
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    return Qdrant.from_documents(
        documents=documents,
        embedding=embeddings,
        location=":memory:",
        collection_name="chatbot"
    )

# Carregar os dados do arquivo Excel
try:
    data = pd.read_excel(file_path)
except FileNotFoundError:
    st.error("O arquivo 'TESTE.xlsx' não foi encontrado. Certifique-se de que está no caminho correto.")
    st.stop()

if "pergunta" in data.columns and "resposta" in data.columns:
    docs = data[["pergunta", "resposta"]]
else:
    st.error("As colunas 'pergunta' e 'resposta' não foram encontradas no arquivo Excel.")
    st.stop()

qdrant = prepare_qdrant(docs)

def custom_prompt(query: str):
    results = qdrant.similarity_search(query, k=8)
    if results:
        source_knowledge = "\n".join(
            [f"Pergunta: {doc.page_content}\nResposta: {doc.metadata['resposta']}" for doc in results]
        )
        augment_prompt = f"""Use o contexto abaixo para responder à pergunta do usuário.
Priorize o uso das informações fornecidas no contexto, mas, se necessário, complemente com seu conhecimento externo.

Contexto:
{source_knowledge}

Pergunta do usuário: {query}"""
    else:
        augment_prompt = f"""A pergunta do usuário não tem informações relevantes disponíveis na base de dados.
Responda com base no seu conhecimento externo de maneira clara e útil.

Pergunta do usuário: {query}"""

    return augment_prompt

query = st.text_input("Digite sua pergunta:")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("Enviar"):
    if query.strip():
        prompt_message = {"role": "user", "content": custom_prompt(query)}
        messages = [
            {"role": "system", "content": "Você é um assistente útil que responde perguntas com base no contexto fornecido e no seu conhecimento externo."},
            prompt_message
        ]
        response = chat.invoke(messages)
        st.session_state.chat_history.insert(0, {"pergunta": query, "resposta": response.content})
    else:
        st.warning("Por favor, insira uma pergunta.")

for entry in st.session_state.chat_history:
    st.markdown(f"<div class='question'><strong>Pergunta:</strong> {entry['pergunta']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='answer'><strong>Resposta:</strong> {entry['resposta']}</div>", unsafe_allow_html=True)