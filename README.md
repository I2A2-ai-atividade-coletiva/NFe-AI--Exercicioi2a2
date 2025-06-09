# NFe-AI - Chat com Documentos Fiscais

Este projeto permite consultar e analisar dados de Notas Fiscais eletrônicas (NFe) e documentos PDF de forma interativa, usando inteligência artificial e um chat em linguagem natural. Desenvolvido em Python com Streamlit e integração com IA via LangChain e Groq.

## 🔀 Versões da Aplicação

O projeto possui **duas versões independentes** do app para consulta dos documentos:

- **app.py**: Utiliza FAISS para indexação vetorial local, embeddings via Hugging Face e permite consultas sem depender de LLMs externas (exceto embeddings). Ideal para uso local e offline (com modelo de embedding disponível).
- **app_groq.py**: Utiliza a API da Groq para consultas com modelos de linguagem de última geração (LLMs) hospedados na nuvem. Necessita de chave de API válida e conexão com a internet.

Escolha a versão conforme sua necessidade de processamento local ou uso de IA em nuvem.

## 📁 Estrutura do Projeto

```
NFe-AI-Exercicioi2a2/
├── .env.exemple        # Exemplo de variáveis de ambiente
├── README.md           # Este arquivo
├── app.py              # App com FAISS + embeddings Hugging Face (consulta local)
├── app_groq.py         # App com API Groq (consulta via LLM na nuvem)
├── requirements.txt    # Dependências do projeto
├── data/               # Dados usados pela aplicação
│   └── 202401_NFs/
│       ├── 202401_NFs_Cabecalho.csv    # CSV com cabeçalhos das NFs
│       └── 202401_NFs_Itens.csv        # CSV com itens das NFs
```

## 🚀 Como Instalar

1. **Clone o repositório:**
   ```bash
   git clone <url-do-repositorio>
   cd NFe-AI-Exercicioi2a2
   ```

2. **Crie um ambiente virtual (opcional, mas recomendado):**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Configuração

1. **Variáveis de ambiente:**
   - Copie o arquivo `.env.exemple` para `.env`:
     ```bash
     cp .env.exemple .env
     ```
   - Preencha os valores necessários, principalmente:
     - `GROQ_API_KEY`: Chave da API Groq (obrigatória)
     - `EXTRACT_DIR`: Caminho para a pasta de dados (opcional, padrão: `data`)
     - `VECTOR_INDEX_PATH`: Caminho para o índice vetorial FAISS (opcional)

2. **Adicione seus arquivos CSV e PDFs na pasta `data/` ou subpastas, conforme desejado.**

## 🏃 Como Rodar

Execute a aplicação principal com Streamlit:

```bash
streamlit run app.py
```

Acesse o endereço exibido no terminal (geralmente http://localhost:8501).

## 💬 Exemplos de Uso

- Pergunte: `Qual o valor total da nota fiscal 12345?`
- Pergunte: `Resuma o documento XYZ.pdf.`
- Pergunte: `Liste todos os produtos vendidos para o cliente João.`

A resposta será gerada pela IA com base nos dados dos arquivos CSV e PDFs presentes em `data/`.

## 📝 Observações

- A aplicação depende de uma chave válida da API Groq para funcionar.
- Os arquivos CSV devem seguir o padrão dos exemplos fornecidos.
- O índice vetorial FAISS é criado automaticamente na primeira execução.
- Para adicionar novos dados, basta colocar os arquivos na pasta `data/` e reiniciar a aplicação.

## 📚 Tecnologias Utilizadas
- Python 3.8+
- Streamlit
- pandas
- langchain
- Groq API
- FAISS
- dotenv

---

Sinta-se à vontade para contribuir ou sugerir melhorias!
