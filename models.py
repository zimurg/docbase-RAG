#models.py

import os, subprocess, sys, json
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def get_model_dir(model_name):
    return os.path.join(os.getenv("OLLAMA_MODEL_DIR", "~/.ollama"), "models", model_name) # TODO: no dejar nada hardcoded


def download_model(model_name):
    model_dir = get_model_dir(model_name)

    #TODO: introducir un check para que se pueda reescribir el modelo si el usuario quiere (ej. en caso de fallar la descarga)

    subprocess.run(["ollama", "pull", model_name], capture_output=True, text=True)

    #TODO: garbage cleaning del directorio después de los errores.


def load_model_config(model_name):
    model_dir = get_model_dir(model_name)
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(model_dir):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)


def load_models(llm_model_name,
                embedding_model_name,
                llm_port=None,
                embed_port=None,
                llm_kwargs={},
                embed_kwargs={}):


    model_dir = get_model_dir(llm_model_name)
    if not os.path.exists(model_dir):
        download_model(llm_model_name)

    embed_dir = get_model_dir(embedding_model_name)
    if not os.path.exists(embed_dir):
        download_model(embedding_model_name)

    if os.path.exists(os.path.join(model_dir, "config.json")) and llm_kwargs=={}:
        llm_kwargs = load_model_config(llm_model_name)
    else:
        pass

    if os.path.exists(os.path.join(embed_dir, "config.json")) and embed_kwargs=={}:
        embed_kwargs = load_model_config(embedding_model_name)
    else:
        pass


    llm = Ollama(
        model=llm_model_name,
        request_timeout=60.0,
        ollama_additional_kwargs=llm_kwargs
        )

    embed_model = HuggingFaceEmbedding(model_name=embedding_model_name, trust_remote_code=True)

    return llm, embed_model
