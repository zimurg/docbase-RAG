#init.py

import argparse, sys, subprocess
from environment import check_and_install_dependencies, check_and_install_ollama


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Herramienta de QA y resumen de documentos')
    parser.add_argument("--update", action="store_true", help="Actualizar las dependencias.")
    parser.add_argument('--data_dir', type=str, default='Data', help='Directorio que contiene los documentos')
    parser.add_argument('--prompt', type=str, help='La consulta a realizar', default='Resume la documentacion')
    parser.add_argument('--files', type=str, nargs='+', default=None, help='Archivos específicos a usar (relativos a data_dir)')
    parser.add_argument('--summarize', action='store_true', help='Resumir la documentación')
    #TODO: permitir la selección de modelos y sus hiperparámetros.
    #TODO: usar explícitamente los metadatos. Crear metadatos de los ficheros sin ellos, o para ampliar los metadatos existentes, haciendo que en cada pasada se identifique si cada fichero lleva sus metadatos y los cree de no ser así. Crear un argumento que permita hacer opt-out de esta funcionalidad.

    args = parser.parse_args()

    check_and_install_ollama()
    check_and_install_dependencies()

    from core import update_docs, create_vector_store, query_engine
    from models import load_models
    from utils import load_config
    from environment import setup_environment

    config = load_config(config_path="config.json")
    data_dir = args.data_dir if args.data_dir is not None else config["data_dir"]
    hidden_prompt = config["hidden_prompt"]
    disclaimer = config["disclaimer"]
    llm_name = config["llm"]
    embeddings_name = config["embeddings_model"]
    #TODO: cargar otros parámetros de configuración, ej. puertos, hiperparámetros etc.

    if args.update:
        update_dependencies()


    llm_config = {} #TODO: cambiar para poder programar los hiperparámetros según el modelo (averiguar como extraer las fichas de Ollama y volcarlas en ficheros json)
    embed_config = {}


    llm, embeddings_model = load_models(llm_name,
                                        embeddings_name,
                                        llm_kwargs=llm_config,
                                        embed_kwargs=embed_config
                                        )

    setup_environment(llm, embeddings_model)


    documents = update_docs(data_dir=args.data_dir,
                            files=args.files)

    if not documents:
        print("No se cargaron documentos. Saliendo.")
        sys.exit(1)

    index = create_vector_store(documents, embeddings_model)
    if not index:
        print("No se logró crear el índice de vectores. Saliendo.")
        sys.exit(1)


    query_engine(index,
                 hidden_prompt,
                 disclaimer,
                 prompt=args.prompt,
                 summarize=args.summarize
                 )
