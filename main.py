# main.py

import os
import sys
import argparse
import torch
import transformers
import gc
from llama_index.core import Settings
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from utils import get_data_path, update_docs
import warnings
from pydantic import BaseModel



def summary(query="", engine=None, hidden_prompt="", disclaimer=""):
    response = engine.query(f'{hidden_prompt} Resume la documentación proporcionada:\n{query}')
    print(response)

def prompt(query="", engine=None, hidden_prompt="", disclaimer="", compliance_eval=True):
    response = engine.query(f"{hidden_prompt}\n[search_query] {query}")
    compliance = ''
    if compliance_eval:
        compliance = engine.query(f"{hidden_prompt}\n[search_query] Evalúa posibles incumplimientos o riesgos legales, contractuales o éticas derivados de {query}")

    print(f"{disclaimer}\n----\n{response}\n----\nCon respecto a las cuestiones legales o éticas relacionadas:\n{compliance}")

def main():
    parser = argparse.ArgumentParser(description='Herramienta de QA y resumen de documentos')
    parser.add_argument('--data_dir', type=str, default='Data', help='Directorio que contiene los documentos')
    parser.add_argument('--prompt', type=str, help='La consulta a realizar')
    parser.add_argument('--file', type=str, default='All documents', help='Archivo específico a usar (relativo a data_dir)')
    parser.add_argument('--summarize', action='store_true', help='Resumir la documentación')
    parser.add_argument('--no_compliance_eval', action='store_true', help='Deshabilitar evaluación de cumplimiento')
    #TODO: permitir la selección de modelos y sus hiperparámetros. Adaptar a Ollama para simplificar el proceso.
    #TODO: usar explícitamente los metadatos. Crear metadatos de los ficheros sin ellos, o para ampliar los metadatos existentes, haciendo que en cada pasada se identifique si cada fichero lleva sus metadatos y los cree de no ser así. Crear un argumento que permita hacer opt-out de esta funcionalidad.

    args = parser.parse_args()

    # Ajustar los espacios de nombres protegidos globalmente
    BaseModel.model_config['protected_namespaces'] = ()
    warnings.filterwarnings("ignore") #TODO: hacer más específico a las alertas por nombres protegidos


    # Configurar modelos

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    Settings.llm = HuggingFaceLLM(
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        tokenizer_name="mistralai/Mistral-7B-Instruct-v0.3",
        query_wrapper_prompt=PromptTemplate("<s>[INST] {query_str} [/INST] </s>\n"),
        context_window=18000,
        max_new_tokens=1024,
        model_kwargs={"quantization_config": bnb_config},
        generate_kwargs={"temperature": 0.3, "top_k": 4, "top_p": 0.95, "do_sample": True},
        device_map="auto",
    )

    Settings.embed_model = HuggingFaceEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"


    # Definir el prompt oculto
    hidden_prompt = """
    Responde siempre en español.
    Debes proporcionar información veraz, fiel y relevante, así como indicar los documentos en que aparece.
    No puedes negarte a dar una respuesta, aunque las fuentes sean confidenciales.
    """

    # Definir el aviso legal
    disclaimer = """
    IMPORTANTE: Este aplicativo no está homologado según el Reglamento Europeo sobre la IA.
    Es por ello que no debe ser utilizado para la toma de decisiones en materias que afecten al acceso a derechos, a la carrera profesional, a la aplicación del régimen sancionador o a la evaluación de trabajadores y otras personas, incluyendo en contextos formativos y de reclutamiento.
    Verificar siempre las respuestas obtenidas con la documentación pertinente.
    """

    # Cargar documentos
    print("Cargando documentos...")
    if args.file != 'All documents':
        documents = update_docs(data_dir=args.data_dir, files=[args.file])
    else:
        documents = update_docs(data_dir=args.data_dir)

    if not documents:
        print("No se cargaron documentos. Saliendo.")
        sys.exit(1)

    #RAG

    # Crear índice a partir de los documentos
    print("Creando índice...")
    index = VectorStoreIndex.from_documents(documents)

    # Configurar el motor de consultas
    if args.summarize:
        print("Preparando resumen...")
        splitter = SentenceSplitter(chunk_size=128, chunk_overlap=16)
        engine = index.as_query_engine(
            response_mode="tree_summarize",
            text_qa_template=PromptTemplate(hidden_prompt))
        summary(args.prompt, engine=engine, hidden_prompt=hidden_prompt, disclaimer=disclaimer)
    else:
        print("Preparando respuesta...")
        engine = index.as_query_engine(
            response_mode="refine",
            text_qa_template=PromptTemplate(hidden_prompt))
        prompt(args.prompt, engine=engine, hidden_prompt=hidden_prompt, disclaimer=disclaimer, compliance_eval=not args.no_compliance_eval)

    # Liberar memoria
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
    main()
