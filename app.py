import os
import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
import requests
import re
from datetime import datetime
import shutil
import logging

# Configura o logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verifica dependências opcionais
try:
    from PyPDF2 import PdfReader
    PDF_PYPDF2_AVAILABLE = True
except ImportError:
    PDF_PYPDF2_AVAILABLE = False

try:
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain.schema import Document
    from langchain_community.vectorstores import FAISS  
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_groq import ChatGroq
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# ----------------------------
# 1. CARREGA VARIÁVEIS DE AMBIENTE
# ----------------------------
load_dotenv()

# ----------------------------
# 2. CONFIGURAÇÕES GLOBAIS
# ----------------------------
DATA_DIR = os.getenv("DATA_DIR", "data")
EXTRACT_DIR = os.getenv("EXTRACT_DIR", DATA_DIR)
VECTOR_INDEX_PATH = os.getenv("VECTOR_INDEX_PATH", os.path.join(DATA_DIR, "faiss_index"))
FAISS_INDEX_FILE = "index.faiss"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Esta variável agora é usada para encontrar o modelo na pasta local "models"
HUGGINGFACE_EMBEDDING_MODEL = os.getenv(
    "HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ----------------------------
# 3. FUNÇÕES UTILITÁRIAS
# ----------------------------
def verificar_ambiente() -> Dict[str, bool]:
    """Verifica a disponibilidade de dependências e configurações"""
    return {
        "groq_api_key": bool(GROQ_API_KEY),
        "pypdf2": PDF_PYPDF2_AVAILABLE,
        "langchain": LANGCHAIN_AVAILABLE,
        "data_dir_exists": os.path.exists(DATA_DIR),
        "extract_dir_exists": os.path.exists(EXTRACT_DIR)
    }

def criar_diretorios():
    """Cria os diretórios necessários se não existirem"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    os.makedirs(VECTOR_INDEX_PATH, exist_ok=True)

# ----------------------------
# 4. FUNÇÕES DE PROCESSAMENTO DE DADOS
# ----------------------------
# (As funções de processamento de dados como carregar_documentos_pdf, etc., permanecem as mesmas)
def carregar_documentos_pdf_pypdf2(diretorio: str) -> List[Dict[str, Any]]:
    documentos_pdf = []
    if not PDF_PYPDF2_AVAILABLE: return documentos_pdf
    if not os.path.exists(diretorio): return documentos_pdf
    st.info(f"Procurando arquivos PDF em: {diretorio}")
    try:
        arquivos_pdf = [f for f in os.listdir(diretorio) if f.lower().endswith(".pdf")]
        for nome_arquivo in arquivos_pdf:
            caminho_completo = os.path.join(diretorio, nome_arquivo)
            try:
                with st.spinner(f"Processando PDF (PyPDF2): {nome_arquivo}..."):
                    reader = PdfReader(caminho_completo)
                    texto_completo = "".join(page.extract_text() for page in reader.pages if page.extract_text())
                    if texto_completo.strip():
                        documentos_pdf.append({"tipo": "pdf", "arquivo": nome_arquivo, "conteudo": texto_completo})
            except Exception as e:
                st.warning(f"Não foi possível ler o arquivo PDF {nome_arquivo}. Erro: {e}")
    except Exception as e:
        st.error(f"Erro ao acessar diretório {diretorio}: {e}")
    return documentos_pdf

def carregar_documentos_pdf_langchain(diretorio: str) -> List[Document]:
    documentos_pdf = []
    if not LANGCHAIN_AVAILABLE: return documentos_pdf
    if not os.path.exists(diretorio): return documentos_pdf
    st.info(f"Procurando arquivos PDF em: {diretorio}")
    try:
        arquivos_pdf = [f for f in os.listdir(diretorio) if f.lower().endswith(".pdf")]
        for nome_arquivo in arquivos_pdf:
            caminho_completo = os.path.join(diretorio, nome_arquivo)
            try:
                with st.spinner(f"Processando PDF (LangChain): {nome_arquivo}..."):
                    loader = PyMuPDFLoader(caminho_completo)
                    documentos_pdf.extend(loader.load())
            except Exception as e:
                st.warning(f"Não foi possível ler o arquivo PDF {nome_arquivo} com PyMuPDF. Erro: {e}")
    except Exception as e:
        st.error(f"Erro ao acessar diretório {diretorio}: {e}")
    return documentos_pdf

def carregar_dataframes() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    csv_dir = os.path.join(EXTRACT_DIR, "202401_NFs")
    cab_path = os.path.join(csv_dir, "202401_NFs_Cabecalho.csv")
    it_path = os.path.join(csv_dir, "202401_NFs_Itens.csv")
    df_cab, df_it = None, None
    try:
        if os.path.isfile(cab_path) and os.path.isfile(it_path):
            st.info("Arquivos CSV de notas fiscais encontrados.")
            df_cab = pd.read_csv(cab_path, sep=",", encoding="utf-8")
            df_it = pd.read_csv(it_path, sep=",", encoding="utf-8")
        else:
            st.info("Arquivos CSV de notas fiscais não encontrados.")
    except Exception as e:
        st.error(f"Erro ao carregar CSVs: {e}")
    return df_cab, df_it

def processar_dados_csv_simples(df_cab, df_it) -> List[Dict[str, Any]]:
    # (Esta função permanece a mesma)
    documentos_csv = []
    if df_cab is not None:
        for _, row in df_cab.iterrows():
            documentos_csv.append({"tipo": "nf_cabecalho", "conteudo": f"CABEÇALHO NF: {row.to_dict()}", "metadata": row.to_dict()})
    if df_it is not None:
        for _, row in df_it.iterrows():
            documentos_csv.append({"tipo": "nf_item", "conteudo": f"ITEM NF: {row.to_dict()}", "metadata": row.to_dict()})
    return documentos_csv
    
def processar_dados_csv_langchain(df_cab, df_it) -> List[Document]:
    # (Esta função permanece a mesma)
    docs = []
    if df_cab is not None:
        for _, row in df_cab.iterrows():
            content = f"Nota Fiscal: {row.get('NÚMERO', '')}. Data: {row.get('DATA EMISSÃO', '')}. Emitente: {row.get('RAZÃO SOCIAL EMITENTE', '')}. Destinatário: {row.get('NOME DESTINATÁRIO', '')}. Valor: R$ {row.get('VALOR NOTA FISCAL', 0):.2f}."
            docs.append(Document(page_content=content, metadata=row.to_dict()))
    if df_it is not None:
        for _, row in df_it.iterrows():
            content = f"Item da NF {row.get('NÚMERO', '')}: Produto {row.get('NÚMERO PRODUTO', '')} - {row.get('DESCRIÇÃO DO PRODUTO/SERVIÇO', '')}. Qtd: {row.get('QUANTIDADE', 0)}. Valor: R$ {row.get('VALOR TOTAL', 0):.2f}."
            docs.append(Document(page_content=content, metadata=row.to_dict()))
    return docs

# ----------------------------
# 5. FUNÇÕES DE IA
# (As funções de IA como consultar_groq, etc., permanecem as mesmas)
# ----------------------------
def buscar_documentos_relevantes(pergunta: str, documentos: List[Dict[str, Any]], max_docs: int = 5) -> List[Dict[str, Any]]:
    # ... (código inalterado) ...
    pergunta_lower = pergunta.lower()
    palavras_chave = set(re.findall(r'\b\w{3,}\b', pergunta_lower))
    docs_com_score = []
    for doc in documentos:
        conteudo_lower = doc['conteudo'].lower()
        score = sum(conteudo_lower.count(palavra) for palavra in palavras_chave)
        if score > 0:
            docs_com_score.append((doc, score))
    docs_com_score.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in docs_com_score[:max_docs]]

def consultar_groq(pergunta: str, contexto: str) -> str:
    # ... (código inalterado) ...
    if not GROQ_API_KEY: return "Erro: Chave da API GROQ não configurada"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    prompt = f"Contexto: {contexto}\n\nPergunta: {pergunta}\n\nResposta:"
    payload = {"model": GROQ_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0}
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Erro na API GROQ: {e}"

# ----------------------------
# 6. FUNÇÕES LANGCHAIN (VERSÃO AVANÇADA) - CÓDIGO AJUSTADO AQUI
# ----------------------------
@st.cache_resource
def criar_ou_carregar_index(_docs: List[Document]) -> Optional[FAISS]:
    """Cria ou carrega o índice FAISS, usando um modelo de embedding local."""
    if not LANGCHAIN_AVAILABLE:
        return None
    
    try:
        # Deriva o caminho local do modelo a partir da variável global para manter a consistência
        local_model_path = os.path.join("models", HUGGINGFACE_EMBEDDING_MODEL)

        # Passo 1: Verificar se a pasta do modelo local existe
        if not os.path.isdir(local_model_path):
            st.error(f"Modelo local de embedding não encontrado em '{local_model_path}'.")
            st.info("Por favor, execute o script 'download_model.py' uma única vez para baixar o modelo e tente novamente.")
            st.stop()  # Interrompe a execução se o modelo não existir

        # Passo 2: Usar o caminho local para carregar os embeddings
        embeddings = HuggingFaceEmbeddings(model_name=local_model_path)

        index_file_path = os.path.join(VECTOR_INDEX_PATH, FAISS_INDEX_FILE)
        
        # Passo 3: Carregar ou criar o índice FAISS
        if os.path.exists(VECTOR_INDEX_PATH) and os.path.isfile(index_file_path):
            st.info("Carregando índice FAISS existente...")
            vectordb = FAISS.load_local(
                VECTOR_INDEX_PATH, 
                embeddings=embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            with st.spinner("Criando novo índice vetorial. Isso pode levar alguns minutos..."):
                if os.path.isdir(VECTOR_INDEX_PATH):
                    shutil.rmtree(VECTOR_INDEX_PATH)
                os.makedirs(VECTOR_INDEX_PATH, exist_ok=True)
                vectordb = FAISS.from_documents(_docs, embedding=embeddings)
                vectordb.save_local(VECTOR_INDEX_PATH)
                
        return vectordb
    
    except Exception as e:
        st.error(f"Erro ao criar/carregar índice vetorial: {e}")
        st.exception(e)  # Mostra detalhes do erro no app para facilitar o debug
        return None

@st.cache_resource
def montar_agente_qa(_vectordb: FAISS) -> Optional[RetrievalQA]:
    # (Esta função permanece a mesma)
    if not LANGCHAIN_AVAILABLE or not _vectordb: return None
    try:
        QA_PROMPT = PromptTemplate(template="Use o seguinte contexto para responder à pergunta. Se não souber a resposta, diga que não encontrou a informação.\n\nContexto:\n{context}\n\nPergunta:\n{question}\n\nResposta:", input_variables=["context", "question"])
        llm = ChatGroq(model_name="llama3-8b-8192", temperature=0, groq_api_key=GROQ_API_KEY)
        retriever = _vectordb.as_retriever(search_kwargs={"k": 5})
        return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt": QA_PROMPT})
    except Exception as e:
        st.error(f"Erro ao montar agente Q&A: {e}")
        return None

# ----------------------------
# 7. INTERFACE STREAMLIT
# ----------------------------
# (As funções da interface como exibir_info_debug, etc., permanecem as mesmas)
def exibir_info_debug():
    # ... (código inalterado) ...
    with st.expander("Informações de Debug"):
        ambiente = verificar_ambiente()
        st.write(ambiente)
        if ambiente['extract_dir_exists']:
            st.write(f"- Conteúdo EXTRACT_DIR: {os.listdir(EXTRACT_DIR)}")

def configurar_api_key():
    # ... (código inalterado) ...
    global GROQ_API_KEY
    if not GROQ_API_KEY:
        st.error("Chave da API Groq não encontrada.")
        temp_key = st.text_input("Insira sua GROQ API Key:", type="password")
        if temp_key:
            GROQ_API_KEY = temp_key
            st.rerun()
        else:
            st.stop()

def carregar_documentos_modo_simples():
    docs_pdf = carregar_documentos_pdf_pypdf2(EXTRACT_DIR)
    df_cab, df_it = carregar_dataframes()
    docs_csv = processar_dados_csv_simples(df_cab, df_it)
    return docs_csv + docs_pdf

def carregar_documentos_modo_avancado():
    docs_pdf = carregar_documentos_pdf_langchain(EXTRACT_DIR)
    df_cab, df_it = carregar_dataframes()
    docs_csv = processar_dados_csv_langchain(df_cab, df_it)
    return docs_csv + docs_pdf

def processar_pergunta_modo_simples(pergunta, documentos):
    # ... (código inalterado) ...
    docs_relevantes = buscar_documentos_relevantes(pergunta, documentos)
    contexto = "\n\n".join([doc['conteudo'] for doc in docs_relevantes]) if docs_relevantes else "Nenhum contexto encontrado."
    resposta = consultar_groq(pergunta, contexto)
    st.markdown("### Resposta:")
    st.write(resposta)

def processar_pergunta_modo_avancado(pergunta, agente_qa):
    # ... (código inalterado) ...
    try:
        resultado = agente_qa.invoke({"query": pergunta})
        st.markdown("### 📝 Resposta:")
        st.write(resultado.get("result", "Não foi possível gerar uma resposta."))
        with st.expander("Ver documentos-fonte utilizados"):
            for doc in resultado.get("source_documents", []):
                st.code(doc.page_content, language='text')
    except Exception as e:
        st.error(f"Erro ao consultar o agente: {e}")

def criar_sidebar():
    # ... (código inalterado) ...
    with st.sidebar:
        st.header("🏛️ Sistema de Análise")
        # Adicione aqui o que mais for útil, como no seu código original

def main():
    st.set_page_config(page_title="Chat com Documentos", layout="wide")
    criar_sidebar()
    st.title("💬 Chat com Documentos Fiscais")
    st.markdown("Faça perguntas em português sobre os dados de suas notas fiscais e documentos PDF.")
    
    criar_diretorios()
    exibir_info_debug()
    configurar_api_key()

    modo_avancado = LANGCHAIN_AVAILABLE and st.checkbox(
        "🚀 Usar modo avançado (LangChain + Vetorização Local)", 
        value=LANGCHAIN_AVAILABLE,
        help="Requer LangChain e um modelo de embedding local. Oferece busca semântica mais precisa."
    )

    documentos = None
    agente_qa = None

    with st.spinner("Analisando e carregando documentos..."):
        if modo_avancado:
            documentos = carregar_documentos_modo_avancado()
            if documentos:
                vector_database = criar_ou_carregar_index(documentos)
                if vector_database:
                    agente_qa = montar_agente_qa(vector_database)
                    st.session_state.agente_qa = agente_qa
                else:
                    st.error("Falha ao inicializar o banco de dados vetorial. O modo avançado não pode continuar.")
                    modo_avancado = False # Fallback para modo simples
        
        if not modo_avancado:
            documentos = carregar_documentos_modo_simples()
            st.session_state.documentos_simples = documentos

    if not documentos:
        st.warning("Nenhum documento foi carregado.")
    else:
        st.success(f"✅ {len(documentos)} blocos de documentos carregados com sucesso!")

    pergunta_usuario = st.text_input("Ex: Qual o valor total da NF 12345?", key="user_question")

    if pergunta_usuario:
        with st.spinner("🔍 Processando sua pergunta..."):
            if modo_avancado and 'agente_qa' in st.session_state:
                processar_pergunta_modo_avancado(pergunta_usuario, st.session_state.agente_qa)
            elif not modo_avancado and 'documentos_simples' in st.session_state:
                processar_pergunta_modo_simples(pergunta_usuario, st.session_state.documentos_simples)
            else:
                st.error("O sistema não está pronto para receber perguntas. Verifique os logs.")

# PONTO DE ENTRADA DA APLICAÇÃO
if __name__ == "__main__":
    main()