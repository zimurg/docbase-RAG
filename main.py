# main.py

import os
import sys
import argparse
import torch
import transformers
import gc
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from utils import get_data_path, update_docs
import warnings
from pydantic import BaseModel, ConfigDict


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
    parser.add_argument('--llm_model_name', type=str, default='mistralai/Mistral-7B-Instruct-v0.3', help='Nombre del modelo LLM')
    parser.add_argument('--embedding_model_name', type=str, default='nomic-ai/nomic-embed-text-v1.5', help='Nombre del modelo de embeddings')
    parser.add_argument('--generate_metadata', action='store_true', help='Generar metadatos para los documentos')  # TODO: cambiar el argumento por uno que permita elegir si en la pasada de procesamiento de los documentos generamos el metadata o hacemos skip cuando no encontramos metadata.
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #TODO: resolver excepción con protected namespaces

    except UserWarning as e:
        print(e)

    # Configurar modelos

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    llm = HuggingFaceLLM(
        model_name=args.llm_model_name,
        tokenizer_name=args.llm_model_name,
        query_wrapper_prompt=PromptTemplate("<s>[INST] {query_str} [/INST] </s>\n"),
        context_window=18000,
        max_new_tokens=1024,
        model_kwargs={"quantization_config": bnb_config},
        generate_kwargs={"temperature": 0.3, "top_k": 4, "top_p": 0.95, "do_sample": True},
        device_map="auto",
    )

    embed_model = HuggingFaceEmbedding(model_name=args.embedding_model_name, trust_remote_code=True)

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

    # Generar metadatos si se especifica. TODO: resolver bug en el que OpenBLAS no asigna correctamente los threads
    if args.generate_metadata:
        print("Generando metadatos...")
        # Los metadatos ya se generaron en update_docs y se almacenaron en doc.extra_info
        # Podemos guardarlos en un archivo JSON si es necesario
        metadata_list = []
        for doc in documents:
            metadata = doc.extra_info
            metadata['doc_id'] = doc.doc_id
            metadata_list.append(metadata)
        # Guardar los metadatos en un archivo JSON
        with open('documents_metadata.json', 'w') as f:
            json.dump(metadata_list, f, indent=4)
        print("Metadatos generados y guardados en documents_metadata.json")
        # Si solo queremos generar metadatos y salir
        sys.exit(0)

    # Filtrar documentos según metadatos (por ejemplo, solo la última versión)
    print("Filtrando documentos según metadatos...")
    latest_docs = {}
    for doc in documents:
        version = doc.extra_info.get('version', '0.0')
        doc_id = doc.doc_id
        # Suponemos que el nombre del documento sin extensión es el identificador
        doc_name = os.path.splitext(doc_id)[0]
        # Comparar versiones y mantener la más reciente
        if doc_name not in latest_docs or version_compare(version, latest_docs[doc_name].extra_info.get('version', '0.0')) > 0:
            latest_docs[doc_name] = doc

    # Actualizar la lista de documentos con las últimas versiones
    documents = list(latest_docs.values())

    # Crear índice a partir de los documentos
    print("Creando índice...")
    index = VectorStoreIndex.from_documents(documents, llm=llm, embed_model=embed_model)

    # Configurar el motor de consultas
    if args.summarize:
        print("Preparando resumen...")
        splitter = SentenceSplitter(chunk_size=128, chunk_overlap=16)
        engine = index.as_query_engine(response_mode="tree_summarize", text_qa_template=PromptTemplate(hidden_prompt))
        summary(args.prompt, engine=engine, hidden_prompt=hidden_prompt, disclaimer=disclaimer)
    else:
        print("Preparando respuesta...")
        engine = index.as_query_engine(response_mode="refine", text_qa_template=PromptTemplate(hidden_prompt))
        prompt(args.prompt, engine=engine, hidden_prompt=hidden_prompt, disclaimer=disclaimer, compliance_eval=not args.no_compliance_eval)

    # Liberar memoria
    torch.cuda.empty_cache()
    gc.collect()

def version_compare(v1, v2):
    # Función para comparar versiones
    # Retorna 1 si v1 > v2, -1 si v1 < v2, 0 si son iguales
    from packaging import version
    return (version.parse(v1) > version.parse(v2)) - (version.parse(v1) < version.parse(v2))

if __name__ == '__main__':
    main()
