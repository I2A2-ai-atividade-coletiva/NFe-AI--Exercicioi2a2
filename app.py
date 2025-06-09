import os
import shutil
import streamlit as st
import pandas as pd
from typing import List
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader

# ----------------------------
# 1. CARREGA VARIÁVEIS DE AMBIENTE
# ----------------------------
load_dotenv()

# ----------------------------
# 2. CONFIGURAÇÕES GLOBAIS
# ----------------------------
DATA_DIR = os.getenv("DATA_DIR", "data")
# A LINHA ABAIXO É A MAIS IMPORTANTE:
# EXTRACT_DIR deve ser igual a DATA_DIR para que o código procure PDFs na pasta 'data'
# e os CSVs na subpasta 'data/202401_NFs'.
EXTRACT_DIR = os.getenv("EXTRACT_DIR", DATA_DIR) 

VECTOR_INDEX_PATH = os.getenv("VECTOR_INDEX_PATH", os.path.join(DATA_DIR, "faiss_index"))
FAISS_INDEX_FILE = "index.faiss"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACE_EMBEDDING_MODEL = os.getenv(
    "HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
# ----------------------------
# 3. FUNÇÕES DE PROCESSAMENTO DE DADOS
# ----------------------------
def carregar_documentos_pdf(diretorio: str) -> List[Document]:
    documentos_pdf = []
    st.info(f"Procurando arquivos PDF em: {diretorio}")
    for nome_arquivo in os.listdir(diretorio):
        if nome_arquivo.lower().endswith(".pdf"):
            caminho_completo = os.path.join(diretorio, nome_arquivo)
            try:
                with st.spinner(f"Processando PDF: {nome_arquivo}..."):
                    loader = PyMuPDFLoader(caminho_completo)
                    documentos_pdf.extend(loader.load())
            except Exception as e:
                st.warning(f"Não foi possível ler o arquivo PDF {nome_arquivo}. Erro: {e}")
    if documentos_pdf:
        st.info(f"Foram encontrados e processados {len(documentos_pdf)} páginas de arquivos PDF.")
    return documentos_pdf

def carregar_dataframes() -> (pd.DataFrame, pd.DataFrame):
    csv_dir = os.path.join(EXTRACT_DIR, "202401_NFs")
    cab_path = os.path.join(csv_dir, "202401_NFs_Cabecalho.csv")
    it_path = os.path.join(csv_dir, "202401_NFs_Itens.csv")
    df_cab, df_it = None, None
    if os.path.isfile(cab_path) and os.path.isfile(it_path):
        st.info("Arquivos CSV de notas fiscais encontrados. Carregando...")
        df_cab = pd.read_csv(cab_path, sep=",", encoding="utf-8")
        df_it = pd.read_csv(it_path, sep=",", encoding="utf-8")
    else:
        st.warning("Arquivos CSV de notas fiscais não encontrados. A busca será feita apenas nos PDFs (se existirem).")
    return df_cab, df_it

def df_to_documents(df_cab: pd.DataFrame, df_it: pd.DataFrame) -> List[Document]:
    docs = []
    if df_cab is not None:
        for _, row in df_cab.iterrows():
            content = (f"Nota Fiscal: {row.get('NÚMERO', '')}. Data de Emissão: {row.get('DATA EMISSÃO', '')}. Emitente: {row.get('RAZÃO SOCIAL EMITENTE', '')}. Destinatário: {row.get('NOME DESTINATÁRIO', '')}. Valor Total da Nota: R$ {row.get('VALOR NOTA FISCAL', 0):.2f}.")
            docs.append(Document(page_content=content, metadata=row.to_dict()))
    if df_it is not None:
        for _, row in df_it.iterrows():
            content = (f"Item da Nota Fiscal {row.get('NÚMERO', '')}: Produto {row.get('NÚMERO PRODUTO', '')} - {row.get('DESCRIÇÃO DO PRODUTO/SERVIÇO', '')}. Quantidade: {row.get('QUANTIDADE', 0)}. Valor Total do Item: R$ {row.get('VALOR TOTAL', 0):.2f}.")
            docs.append(Document(page_content=content, metadata=row.to_dict()))
    return docs

def carregar_todos_os_documentos(diretorio_principal: str) -> List[Document]:
    docs_pdf = carregar_documentos_pdf(diretorio_principal)
    df_cab, df_it = carregar_dataframes()
    docs_csv = df_to_documents(df_cab, df_it)
    return docs_csv + docs_pdf

# ----------------------------
# 4. FUNÇÕES DE IA E LANGCHAIN
# ----------------------------
@st.cache_resource
def criar_ou_carregar_index(_docs: List[Document]) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=HUGGINGFACE_EMBEDDING_MODEL)
    index_file_path = os.path.join(VECTOR_INDEX_PATH, FAISS_INDEX_FILE)
    if os.path.exists(VECTOR_INDEX_PATH) and os.path.isfile(index_file_path):
        st.info("Carregando índice FAISS existente...")
        vectordb = FAISS.load_local(VECTOR_INDEX_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)
    else:
        with st.spinner("Criando um novo índice vetorial com todos os documentos. Isso pode levar alguns minutos..."):
            if os.path.isdir(VECTOR_INDEX_PATH):
                shutil.rmtree(VECTOR_INDEX_PATH)
            os.makedirs(VECTOR_INDEX_PATH, exist_ok=True)
            vectordb = FAISS.from_documents(_docs, embedding=embeddings)
            vectordb.save_local(VECTOR_INDEX_PATH)
    return vectordb

@st.cache_resource
def montar_agente_qa(_vectordb: FAISS) -> RetrievalQA:
    # AQUI ESTÁ A CORREÇÃO PRINCIPAL
    QA_PROMPT = PromptTemplate(
        template=(
            "Você é um assistente especialista em análise de dados de Notas Fiscais e documentos.\n"
            "Use estritamente o contexto fornecido para responder à pergunta do usuário de forma clara e objetiva.\n"
            "Se a informação não estiver no contexto, responda: 'Não encontrei informações sobre isso nos documentos fornecidos.'\n\n"
            "Contexto:\n{context}\n\n"
            "Pergunta:\n{question}\n\n"
            "Resposta:"
        ),
        input_variables=["context", "question"]
    )
    
    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0, groq_api_key=GROQ_API_KEY)
    retriever = _vectordb.as_retriever(search_kwargs={"k": 4})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt": QA_PROMPT})
    return qa_chain

# ----------------------------
# 5. INTERFACE STREAMLIT
# ----------------------------
st.set_page_config(page_title="Chat com Documentos", layout="wide", initial_sidebar_state="collapsed")
st.title("📄 Chat com Documentos (CSVs e PDFs)")
st.markdown("Faça perguntas em português sobre os dados das suas notas fiscais e documentos PDF.")

if not GROQ_API_KEY:
    st.error("A chave da API da Groq (GROQ_API_KEY) não foi encontrada.")
    st.info("Por favor, crie um arquivo .env na raiz do projeto e adicione a sua chave.")
    st.stop()

try:
    documentos_unificados = carregar_todos_os_documentos(EXTRACT_DIR)
    if not documentos_unificados:
        st.warning("Nenhum documento (CSV ou PDF) foi encontrado ou pôde ser lido. A aplicação não pode continuar.")
        st.stop()
    vector_database = criar_ou_carregar_index(_docs=documentos_unificados)
    agente_qa = montar_agente_qa(vector_database)
    st.success("✅ Ambiente pronto! Todos os documentos foram carregados e indexados.")
except Exception as e:
    st.error(f"❌ Ocorreu um erro durante a inicialização: {e}")
    st.stop()

st.header("Faça sua pergunta:")
pergunta_usuario = st.text_input("Ex: Qual o valor total da nota fiscal 12345? ou Resuma o documento XYZ.pdf", key="user_question")

if pergunta_usuario:
    with st.spinner("Consultando a Groq e os documentos... ⚡️"):
        try:
            resultado = agente_qa.invoke({"query": pergunta_usuario})
            resposta = resultado.get("result", "Não foi possível gerar uma resposta.")
            fontes = resultado.get("source_documents", [])
            st.markdown("### 📝 Resposta:")
            st.write(resposta)
            if fontes:
                with st.expander("Ver documentos-fonte utilizados"):
                    for doc in fontes:
                        st.code(doc.page_content, language='text')
                        st.json(doc.metadata)
                        st.markdown("---")
        except Exception as e:
            st.error(f"❌ Erro ao consultar o agente: {e}")