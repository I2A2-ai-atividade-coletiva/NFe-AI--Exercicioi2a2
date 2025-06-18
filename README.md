# NFe-AI - Chat com Documentos Fiscais

Este projeto oferece uma interface de chat interativa para consultar e analisar dados de Notas Fiscais eletrônicas (NFe) e documentos PDF usando inteligência artificial. Desenvolvido em Python, a solução integra a velocidade da API Groq com a precisão da busca vetorial local para fornecer respostas rápidas e contextuais.

## 🌟 Principais Funcionalidades

- **Chat Interativo:** Converse com seus documentos em linguagem natural.
- **Suporte a Múltiplos Formatos:** Analisa dados de arquivos CSV e extrai texto de documentos PDF.
- **Dois Modos de Operação:**
    - **Modo Avançado (Padrão):** Utiliza busca vetorial semântica (LangChain + FAISS + Embeddings Locais) para encontrar os trechos mais relevantes dos documentos antes de consultar o LLM. Oferece respostas mais precisas e contextuais.
    - **Modo Simples (Fallback):** Utiliza uma busca por palavra-chave mais leve, ideal para ambientes com menos recursos ou para quando as dependências avançadas não estão disponíveis.
- **Processamento Local:** O modelo de embeddings (que transforma texto em números) e o índice vetorial rodam 100% localmente, garantindo que o conteúdo dos seus documentos não saia da sua máquina (apenas a pergunta e o contexto relevante são enviados ao LLM).

## 📁 Estrutura do Projeto

```
NFe-AI-Exercicioi2a2/
├── .venv/              # Ambiente virtual Python (gerado localmente)
├── data/               # Pasta para colocar seus arquivos CSV e PDF.
│   └── 202401_NFs/
│       ├── 202401_NFs_Cabecalho.csv
│       └── 202401_NFs_Itens.csv
├── models/             # Pasta para modelos Hugging Face (gerado pelo download_model.py).
│   └── sentence-transformers/
│       └── all-MiniLM-L6-v2/
├── .env                # Arquivo com suas variáveis de ambiente (criado a partir do .env.exemple).
├── .env.exemple        # Arquivo de exemplo para as variáveis de ambiente.
├── .gitignore          # Arquivos e pastas a serem ignorados pelo Git.
├── app.py              # Script principal da aplicação Streamlit.
├── download_model.py   # Script para baixar o modelo de embeddings.
├── requirements.txt    # Lista de dependências do projeto.
└── README.md           # Este arquivo.
```

## 🚀 Instalação e Configuração

Siga estes passos para configurar o ambiente e rodar o projeto.

### 1. Clone o Repositório
```bash
git clone https://github.com/I2A2-ai-atividade-coletiva/NFe-AI--Exercicioi2a2
cd NFe-AI-Exercicioi2a2
```

### 2. Crie e Ative um Ambiente Virtual (Recomendado)
O uso de um ambiente virtual é essencial para evitar conflitos de dependências.

```bash
# Cria o ambiente
python -m venv .venv

# Ativa o ambiente
# No Windows:
.venv\Scripts\activate
# No macOS/Linux:
source .venv/bin/activate
```
> **Atenção:** Em sistemas Linux modernos (como Ubuntu 23.04+), a instalação de pacotes com `pip` fora de um ambiente virtual é bloqueada (PEP 668). O uso do ambiente virtual resolve isso.

### 3. Instale as Dependências
Com o ambiente ativado, instale todas as bibliotecas necessárias a partir do `requirements.txt`.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Baixe o Modelo de Embedding
Para que o **Modo Avançado** funcione, você precisa baixar o modelo de embedding localmente. Execute este comando **apenas uma vez**:

```bash
python download_model.py
```
Isso criará a pasta `models` com os arquivos do modelo.

### 5. Configure as Variáveis de Ambiente
Você precisa de uma chave da API Groq para que o LLM funcione.

```bash
# Copie o arquivo de exemplo
cp .env.exemple .env
```
Agora, abra o arquivo `.env` e adicione sua chave da API Groq:

```dotenv
# .env
GROQ_API_KEY="gsk_SuaChaveDaAPIAqui..."
```

### 6. Adicione Seus Dados
Coloque seus arquivos `.csv` e `.pdf` dentro da pasta `data/`. A aplicação buscará os arquivos automaticamente neste diretório.

## 🏃 Como Rodar a Aplicação

Com o ambiente ativado e as configurações prontas, inicie a aplicação Streamlit:

```bash
streamlit run app.py
```

Acesse o endereço local que aparecer no terminal (geralmente `http://localhost:8501`).

## 🛠️ Solução de Erros Comuns (Troubleshooting)

- **Erro `externally-managed-environment`:** Você esqueceu de ativar o ambiente virtual (`source .venv/bin/activate`) antes de usar o `pip install`.

- **Aplicação não mostra tela:** Verifique se não há erros de sintaxe no terminal onde você executou `streamlit run`. Certifique-se também de que a função `main()` é chamada no final do `app.py` com o bloco `if __name__ == "__main__":`.

- **Erro `Modelo local não encontrado`:** Você não executou o script de download. Pare a aplicação e rode `python download_model.py`.

- **Erro de Chave Groq:** Verifique se a chave no arquivo `.env` está correta e se o arquivo foi nomeado exatamente como `.env` (e não `.env.txt`).

- **Arquivos não são encontrados:** Certifique-se de que seus arquivos PDF e CSV estão dentro da pasta `data` na raiz do projeto.

## 📚 Tecnologias Utilizadas

- **Python 3.8+**
- **Streamlit:** Para a interface web interativa.
- **LangChain:** Para orquestrar o fluxo de IA (RAG).
- **Groq API:** Para inferência de LLM de alta velocidade.
- **Sentence-Transformers (Hugging Face):** Para a geração de embeddings de texto.
- **FAISS (Facebook AI):** Para busca de similaridade vetorial de alta performance.
- **Pandas:** Para manipulação de dados de arquivos CSV.
- **PyMuPDF / PyPDF2:** Para extração de texto de arquivos PDF.

