import os
import json
import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
import requests
import re
from datetime import datetime

# Verificar se PyPDF2 está disponível, caso contrário usar alternativa
try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    st.warning("PyPDF2 não está instalado. Funcionalidade de PDF será limitada.")
    PDF_AVAILABLE = False

# ----------------------------
# 1. CARREGA VARIÁVEIS DE AMBIENTE
# ----------------------------
load_dotenv()

# ----------------------------
# 2. CONFIGURAÇÕES GLOBAIS
# ----------------------------
DATA_DIR = os.getenv("DATA_DIR", "data")
EXTRACT_DIR = os.getenv("EXTRACT_DIR", DATA_DIR)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ----------------------------
# 3. FUNÇÕES DE PROCESSAMENTO DE DADOS
# ----------------------------
def carregar_documentos_pdf(diretorio: str) -> List[Dict[str, Any]]:
    """Carrega e extrai texto de arquivos PDF"""
    documentos_pdf = []
    
    if not PDF_AVAILABLE:
        st.info("PyPDF2 não disponível. Pulando processamento de PDFs.")
        return documentos_pdf
    
    st.info(f"Procurando arquivos PDF em: {diretorio}")
    
    if not os.path.exists(diretorio):
        st.warning(f"Diretório {diretorio} não encontrado.")
        return documentos_pdf
    
    try:
        arquivos = os.listdir(diretorio)
        arquivos_pdf = [f for f in arquivos if f.lower().endswith(".pdf")]
        
        if not arquivos_pdf:
            st.info("Nenhum arquivo PDF encontrado.")
            return documentos_pdf
            
        for nome_arquivo in arquivos_pdf:
            caminho_completo = os.path.join(diretorio, nome_arquivo)
            try:
                with st.spinner(f"Processando PDF: {nome_arquivo}..."):
                    reader = PdfReader(caminho_completo)
                    texto_completo = ""
                    
                    for page_num, page in enumerate(reader.pages):
                        try:
                            texto_pagina = page.extract_text()
                            if texto_pagina:
                                texto_completo += f"\n--- Página {page_num + 1} ---\n{texto_pagina}"
                        except Exception as e:
                            st.warning(f"Erro ao extrair página {page_num + 1} de {nome_arquivo}: {e}")
                            continue
                    
                    if texto_completo.strip():
                        documentos_pdf.append({
                            "tipo": "pdf",
                            "arquivo": nome_arquivo,
                            "conteudo": texto_completo,
                            "metadata": {
                                "nome_arquivo": nome_arquivo,
                                "total_paginas": len(reader.pages),
                                "data_processamento": datetime.now().isoformat()
                            }
                        })
                        
            except Exception as e:
                st.warning(f"Não foi possível ler o arquivo PDF {nome_arquivo}. Erro: {e}")
    
    except Exception as e:
        st.error(f"Erro ao acessar diretório {diretorio}: {e}")
        return documentos_pdf
    
    if documentos_pdf:
        st.info(f"Foram encontrados e processados {len(documentos_pdf)} arquivos PDF.")
    
    return documentos_pdf

def carregar_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os arquivos CSV de notas fiscais"""
    csv_dir = os.path.join(EXTRACT_DIR, "202401_NFs")
    cab_path = os.path.join(csv_dir, "202401_NFs_Cabecalho.csv")
    it_path = os.path.join(csv_dir, "202401_NFs_Itens.csv")
    
    df_cab, df_it = None, None
    
    try:
        if os.path.isdir(csv_dir):
            st.info(f"Verificando CSVs em: {csv_dir}")
            
            if os.path.isfile(cab_path) and os.path.isfile(it_path):
                st.info("Arquivos CSV de notas fiscais encontrados. Carregando...")
                df_cab = pd.read_csv(cab_path, sep=",", encoding="utf-8")
                df_it = pd.read_csv(it_path, sep=",", encoding="utf-8")
                st.success(f"CSVs carregados: {len(df_cab)} cabeçalhos, {len(df_it)} itens")
            else:
                st.info("Arquivos CSV específicos não encontrados.")
                st.info(f"Procurando por: {cab_path}")
                st.info(f"Procurando por: {it_path}")
        else:
            st.info(f"Diretório CSV não encontrado: {csv_dir}")
            
    except Exception as e:
        st.error(f"Erro ao carregar CSVs: {e}")
        return None, None
    
    return df_cab, df_it

def processar_dados_csv(df_cab: pd.DataFrame, df_it: pd.DataFrame) -> List[Dict[str, Any]]:
    """Converte DataFrames em documentos estruturados"""
    documentos_csv = []
    
    if df_cab is not None:
        for _, row in df_cab.iterrows():
            conteudo = (
                f"NOTA FISCAL - CABEÇALHO\n"
                f"Número: {row.get('NÚMERO', 'N/A')}\n"
                f"Data de Emissão: {row.get('DATA EMISSÃO', 'N/A')}\n"
                f"Emitente: {row.get('RAZÃO SOCIAL EMITENTE', 'N/A')}\n"
                f"Destinatário: {row.get('NOME DESTINATÁRIO', 'N/A')}\n"
                f"Valor Total: R$ {row.get('VALOR NOTA FISCAL', 0):.2f}"
            )
            
            documentos_csv.append({
                "tipo": "nf_cabecalho",
                "numero_nf": row.get('NÚMERO', 'N/A'),
                "conteudo": conteudo,
                "metadata": row.to_dict()
            })
    
    if df_it is not None:
        for _, row in df_it.iterrows():
            conteudo = (
                f"NOTA FISCAL - ITEM\n"
                f"Número NF: {row.get('NÚMERO', 'N/A')}\n"
                f"Produto: {row.get('NÚMERO PRODUTO', 'N/A')}\n"
                f"Descrição: {row.get('DESCRIÇÃO DO PRODUTO/SERVIÇO', 'N/A')}\n"
                f"Quantidade: {row.get('QUANTIDADE', 0)}\n"
                f"Valor Total do Item: R$ {row.get('VALOR TOTAL', 0):.2f}"
            )
            
            documentos_csv.append({
                "tipo": "nf_item",
                "numero_nf": row.get('NÚMERO', 'N/A'),
                "conteudo": conteudo,
                "metadata": row.to_dict()
            })
    
    return documentos_csv

def carregar_todos_os_documentos(diretorio_principal: str) -> List[Dict[str, Any]]:
    """Carrega todos os documentos (PDFs e CSVs)"""
    docs_pdf = carregar_documentos_pdf(diretorio_principal)
    df_cab, df_it = carregar_dataframes()
    docs_csv = processar_dados_csv(df_cab, df_it)
    
    return docs_csv + docs_pdf

# ----------------------------
# 4. FUNÇÕES DE IA COM GROQ
# ----------------------------
def buscar_documentos_relevantes(pergunta: str, documentos: List[Dict[str, Any]], max_docs: int = 5) -> List[Dict[str, Any]]:
    """Busca simples por relevância usando palavras-chave"""
    pergunta_lower = pergunta.lower()
    palavras_chave = re.findall(r'\b\w+\b', pergunta_lower)
    
    docs_com_score = []
    
    for doc in documentos:
        conteudo_lower = doc['conteudo'].lower()
        score = 0
        
        # Pontuação baseada em palavras-chave
        for palavra in palavras_chave:
            if len(palavra) > 2:  # Ignora palavras muito pequenas
                score += conteudo_lower.count(palavra)
        
        # Bonus para números específicos (como números de NF)
        numeros_pergunta = re.findall(r'\d+', pergunta)
        for numero in numeros_pergunta:
            if numero in doc['conteudo']:
                score += 10
        
        if score > 0:
            docs_com_score.append((doc, score))
    
    # Ordena por score e retorna os mais relevantes
    docs_com_score.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in docs_com_score[:max_docs]]

def consultar_groq(pergunta: str, contexto: str) -> str:
    """Consulta a API do GROQ"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""Você é um assistente especialista em análise de dados de Notas Fiscais e documentos.
Use estritamente o contexto fornecido para responder à pergunta do usuário de forma clara e objetiva.
Se a informação não estiver no contexto, responda: 'Não encontrei informações sobre isso nos documentos fornecidos.'

Contexto:
{contexto}

Pergunta: {pergunta}

Resposta:"""
    
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    
    except requests.exceptions.RequestException as e:
        return f"Erro na comunicação com a API GROQ: {e}"
    except KeyError as e:
        return f"Erro ao processar resposta da API: {e}"
    except Exception as e:
        return f"Erro inesperado: {e}"

# ----------------------------
# 5. INTERFACE STREAMLIT
# ----------------------------
def main():
    """Função principal da aplicação"""
    st.set_page_config(
        page_title="Chatcom Notas Fiscais", 
        layout="wide", 
        initial_sidebar_state="collapsed"
    )

    st.title("Chat com Notas Fiscais")
    st.markdown("Faça perguntas em português sobre os dados das suas notas fiscais e documentos PDF.")

    # Debug: Mostrar informações do ambiente
    with st.expander("Informações de Debug", expanded=False):
        st.write("**Variáveis de ambiente:**")
        st.write(f"- DATA_DIR: {DATA_DIR}")
        st.write(f"- EXTRACT_DIR: {EXTRACT_DIR}")
        st.write(f"- GROQ_API_KEY configurada: {'Sim' if GROQ_API_KEY else 'Não'}")
        st.write(f"- GROQ_MODEL: {GROQ_MODEL}")
        st.write(f"- PyPDF2 disponível: {'Sim' if PDF_AVAILABLE else 'Não'}")
        
        st.write("**Estrutura de diretórios:**")
        if os.path.exists(EXTRACT_DIR):
            st.write(f"Diretório principal existe: {EXTRACT_DIR}")
            try:
                arquivos = os.listdir(EXTRACT_DIR)
                st.write(f"- Conteúdo: {arquivos}")
            except Exception as e:
                st.write(f"Erro ao listar: {e}")
        else:
            st.write(f"Diretório principal não existe: {EXTRACT_DIR}")

    # Verificação da API Key
    if not GROQ_API_KEY:
        st.error("A chave da API da Groq (GROQ_API_KEY) não foi encontrada.")
        st.info("Por favor, crie um arquivo .env na raiz do projeto e adicione: GROQ_API_KEY=sua_chave_aqui")
        
        # Permitir inserir a chave temporariamente
        st.markdown("### Ou insira a chave temporariamente:")
        temp_key = st.text_input("GROQ API Key:", type="password", key="temp_groq_key")
        if temp_key:
            os.environ["GROQ_API_KEY"] = temp_key
            st.success("Chave configurada temporariamente!")
            st.rerun()
        else:
            st.stop()

    # Carregamento dos documentos
    try:
        st.markdown("### Carregando Notas Fiscais...")
        
        with st.spinner("Carregando Notas Fiscais..."):
            documentos_unificados = carregar_todos_os_documentos(EXTRACT_DIR)
        
        if not documentos_unificados:
            st.warning("Nenhum documento (CSV ou PDF) foi encontrado ou pôde ser lido.")
            st.info("**Possíveis soluções:**")
            st.info("1. Verifique se os arquivos estão na pasta correta")
            st.info("2. Verifique as permissões dos arquivos")
            st.info("3. Certifique-se de que os arquivos não estão corrompidos")
            
            # Continuar mesmo sem documentos para teste
            st.session_state.documentos = []
            st.info("Continuando sem documentos para teste da interface...")
        else:
            # Armazenar documentos na sessão
            st.session_state.documentos = documentos_unificados
            st.success(f"{len(documentos_unificados)} Notas Fiscais carregadas com sucesso!")
            
            # Mostrar estatísticas dos documentos
            tipos_docs = {}
            for doc in documentos_unificados:
                tipo = doc['tipo']
                tipos_docs[tipo] = tipos_docs.get(tipo, 0) + 1
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de Documentos", len(documentos_unificados))
            with col2:
                st.metric("Cabeçalhos NF", tipos_docs.get('nf_cabecalho', 0))
            with col3:
                st.metric("Itens NF", tipos_docs.get('nf_item', 0))
            
            if tipos_docs.get('pdf', 0) > 0:
                st.metric("Arquivos PDF", tipos_docs.get('pdf', 0))

    except Exception as e:
        st.error(f"Erro durante o carregamento: {e}")
        st.exception(e)  # Mostra o stack trace completo
        # Continuar mesmo com erro
        st.session_state.documentos = []

    # Interface de perguntas
    st.markdown("### Faça sua pergunta:")
    
    # Verificar se há documentos
    if not hasattr(st.session_state, 'documentos'):
        st.session_state.documentos = []
    
    if len(st.session_state.documentos) == 0:
        st.info("Nenhuma Nota Fiscal carregada. As respostas podem ser limitadas.")
    
    pergunta_usuario = st.text_input(
        "Ex: Qual o valor total da Nota Fiscal 12345? ou Resuma o documento XYZ.pdf", 
        key="user_question",
        placeholder="Digite sua pergunta aqui..."
    )

    if pergunta_usuario:
        with st.spinner("Processando sua pergunta... "):
            try:
                if len(st.session_state.documentos) > 0:
                    # Buscar documentos relevantes
                    docs_relevantes = buscar_documentos_relevantes(
                        pergunta_usuario, 
                        st.session_state.documentos
                    )
                    
                    if not docs_relevantes:
                        st.warning("Não foram encontrados documentos relevantes para sua pergunta.")
                        st.info("Tente reformular a pergunta ou usar termos mais específicos.")
                        # Tentar resposta sem contexto
                        contexto = "Nenhum documento específico encontrado."
                    else:
                        # Preparar contexto
                        contexto = "\n\n".join([doc['conteudo'] for doc in docs_relevantes])
                else:
                    st.info("Respondendo sem documentos carregados...")
                    docs_relevantes = []
                    contexto = "Nenhum documento foi carregado no sistema."
                
                # Consultar GROQ
                resposta = consultar_groq(pergunta_usuario, contexto)
                
                # Exibir resposta
                st.markdown("### Resposta:")
                st.write(resposta)
                
                # Mostrar documentos utilizados (se houver)
                if docs_relevantes:
                    with st.expander(f"Ver {len(docs_relevantes)} documento(s) utilizado(s)"):
                        for i, doc in enumerate(docs_relevantes, 1):
                            st.markdown(f"**Documento {i} - Tipo: {doc['tipo']}**")
                            if doc['tipo'] == 'pdf':
                                st.markdown(f"*Arquivo: {doc.get('arquivo', 'N/A')}*")
                            elif 'numero_nf' in doc:
                                st.markdown(f"*Nota Fiscal: {doc.get('numero_nf', 'N/A')}*")
                            
                            st.text_area(
                                f"Conteúdo {i}:", 
                                doc['conteudo'][:500] + "..." if len(doc['conteudo']) > 500 else doc['conteudo'],
                                height=150,
                                key=f"doc_content_{i}"
                            )
                            st.markdown("---")
                            
            except Exception as e:
                st.error(f"Erro ao processar pergunta: {e}")
                st.exception(e)

    # Sidebar com informações
    with st.sidebar:
        st.header("ℹInformações")
        st.markdown(f"**Modelo:** {GROQ_MODEL}")
        st.markdown(f"**Pasta de dados:** {EXTRACT_DIR}")
        
        if hasattr(st.session_state, 'documentos') and st.session_state.documentos:
            tipos_docs = {}
            for doc in st.session_state.documentos:
                tipo = doc['tipo']
                tipos_docs[tipo] = tipos_docs.get(tipo, 0) + 1
                
            st.markdown("**Documentos carregados:**")
            for tipo, count in tipos_docs.items():
                st.markdown(f"- {tipo}: {count}")
        else:
            st.markdown("**Nenhum documento carregado**")
        
        st.markdown("---")
        st.markdown("**Como usar:**")
        st.markdown("1. Faça perguntas específicas sobre as notas fiscais")
        st.markdown("2. Use números de NF para buscar informações específicas")
        st.markdown("3. Pergunte sobre valores, produtos, emitentes, etc.")
        
        st.markdown("---")
        st.markdown("**Exemplos de perguntas:**")
        st.markdown("- Qual o valor da Nota Fiscal 12345?")
        st.markdown("- Quem é o emitente da Nota Fiscal 67890?")
        st.markdown("- Quais produtos estão na Nota Fiscal 11111?")
        st.markdown("- Resuma o conteúdo do arquivo XYZ.pdf")

# Executar a aplicação
if __name__ == "__main__":
    main()