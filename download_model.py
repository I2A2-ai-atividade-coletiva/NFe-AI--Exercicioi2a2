import os
from sentence_transformers import SentenceTransformer
import sys

# Pega o nome do modelo do .env ou usa o padrão
from dotenv import load_dotenv
load_dotenv()
MODEL_NAME = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# O modelo será salvo em: ./models/sentence-transformers/all-MiniLM-L6-v2
SAVE_PATH = os.path.join("models", MODEL_NAME)

def download_model():
    """
    Baixa e salva um modelo de sentence-transformer do Hugging Face.
    """
    print("-" * 50)
    print(f"Iniciando o download do modelo: {MODEL_NAME}")
    print(f"O modelo será salvo em: {SAVE_PATH}")
    print("-" * 50)

    # Cria o diretório de destino se não existir
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    if os.path.exists(SAVE_PATH):
        print("Diretório do modelo já existe. Pulando o download.")
        return

    try:
        # Instancia o modelo. Isso fará o download dos arquivos necessários.
        model = SentenceTransformer(MODEL_NAME)
        
        # Salva o modelo no formato que a biblioteca pode carregar localmente
        model.save(SAVE_PATH)
        
        print("\n" + "=" * 50)
        print(f"✅ Modelo baixado e salvo com sucesso!")
        print(f"   Local: {os.path.abspath(SAVE_PATH)}")
        print("=" * 50)

    except Exception as e:
        print("\n" + "!" * 50)
        print(f"❌ Falha no download do modelo.")
        print(f"   Erro: {e}")
        print("   Por favor, verifique sua conexão com a internet ou se há um firewall bloqueando o acesso.")
        print("   Você pode tentar executar 'huggingface-cli logout' no seu terminal para limpar credenciais antigas.")
        print("!" * 50)
        sys.exit(1) # Sai com código de erro

if __name__ == "__main__":
    download_model()