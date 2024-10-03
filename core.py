# core.py
import os
from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.readers.file import DocxReader,MarkdownReader,FlatReader
# TODO:desde llama_index>0.6.33, los lectores dependen del módulo llama_hub. Dado que es una iniciativa comunitaria, verificar si cumplen con los requisitos de seguridad. Si existen consideraciones, hacer rollback.
from custom_pdf_reader import CustomPDFReader
from directory_reader import CustomSimpleDirectoryReader
from utils import get_data_path


##################################
# TODO: lógica de preprocesamiento. Queremos preprocesar toda la base sólo cuando sea necesario: la primera vez, de forma programada, o ante cambios significativos. El resto de las veces, procesamos sólo lo que ha cambiado, integrandolo en el index. En el caso de documentos versionados, hay que extraer las versiones obsoletas antes de integrar las vigentes.
###################################

def update_docs(data_dir="Data", files=None):
    data_path = get_data_path(data_dir=data_dir)

    pdf_reader = CustomPDFReader()
    docx_reader = DocxReader()
    txt_reader = FlatReader()
    md_reader = MarkdownReader()

    file_extractor_dict = {
        ".pdf": pdf_reader,
        ".docx": docx_reader,
        ".doc": docx_reader,
        ".txt": txt_reader,
        ".md": md_reader,
        # TODO: agregar más tipos de archivos, como .csv; agregar compatibilidad con bases SQL
    }

    try:
        if files:
            # Procesar archivos específicos
            input_files = [os.path.join(data_path, f) for f in files]
            documents = CustomSimpleDirectoryReader(
                input_files=input_files,
                file_extractor=file_extractor_dict
            ).load_data()
        else:
            # Procesar todo el directorio de datos
            documents = CustomSimpleDirectoryReader(
                input_dir=data_path,
                recursive=True,
                file_extractor=file_extractor_dict
            ).load_data()
            print("Se ha completado la carga")


    except Exception as e:
        print(f"Ocurrió un error al cargar los documentos: {e}")
        documents = []
        raise e

    return documents


def parse_nodes(documents, embed_model):
    parser = SemanticSplitterNodeParser(
        embed_model=embed_model,
        buffer_size=3, #Para capturar relaciones complejas entre conceptos (ej. Quality System) y compensar la tendencia del castellano a introducir determinantes entre medias
        breakpoint_percentile_threshold=95,
        include_metadata=True,
        include_prev_next_rel=True) #embed_model está fijado ya en settings

    nodes = parser.get_nodes_from_documents(documents)

    return nodes


def create_vector_store(documents, embed_model, **kwargs):

    #TODO: Crear objeto SentenceSplitter para pasar como argumento al índice, ej. SentenceSplitter(chunk_size=128, chunk_overlap=8) -> hacer que si no se introduce a mano, se calcule dinámicamente según el tamaño de documents

    #TODO: crear condición para que se cree VectorStoreIndex sólo si no hay uno guardado o si se está leyendo un sólo documento (crear un flag en los argumentos para esto).
    #TODO: guardar el VectorStoreIndex en un fichero con index.storage_context.persist(persist_dir="<persist_dir>"). Cargar con: storage_context = StorageContext.from_defaults(persist_dir="<persist_dir>") e: index = load_index_from_storage(storage_context)
    #TODO: crear una estrategia de extracción de palabras clave (ej. TF-IDF, ner) e introducir como metadato. Crear niveles de jerarquía y referencias cruzadas como metadatos. Crear índice que permita relaciones complejas (GraphIndex, TreeIndex, CompositionalGraph, GPTTreeIndex...)
    print("Procesando nodos...")
    nodes = parse_nodes(documents, embed_model)

    print("Creando índice...")
    index = VectorStoreIndex(nodes=nodes, **kwargs)

    #TODO: VectorStoreInfo para cuando existan varios vectorstore (ej, por fecha de creación)

    return index

def query_engine(index, hidden_prompt, disclaimer, prompt="", summarize=False, top_k=None, **kwargs):
    #TODO: se pueden pasar kwargs específicos desde aquí a los nodos o la vector_store, como filtros de metadatos. Esta sería una manera de resolver el permisionado de usuarios, o de hacer consultas sólo sobre un nivel de la documentación (ej. para nodos: filters=MetadataFilters(filters=[ExactMatchFilter(key='name', value='xxxx')])).

    top_k = top_k if top_k is not None else 3


    if summarize:
        print("Preparando resumen...")
        engine = index.as_query_engine(
            response_mode="tree_summarize",
            text_qa_template=PromptTemplate(hidden_prompt),
            similarity_top_k=top_k,
        )
    else:
        print("Preparando respuesta...")
        engine = index.as_query_engine(
            response_mode="refine",
            text_qa_template=PromptTemplate(hidden_prompt),
            similarity_top_k=top_k,
        )

    #  TODO: existe la funcionalidad para leer de varios indexes; es posible de esa manera implementar permisionados a ese nivel, o crear grupos de documentos según su periodicidad de cambio, de forma que no haya que reconstruir todo el index cada vez que cambien esos documentos.
    #TODO: mejorar la "transfersal logic"; la logica agencial con la que el engine explora el índice. Son estrategias de búsqueda, como breadth-first, depth-first, relevance-guided (similarity score)... Puede introducirse como kwarg en el engine, ej:traversal_kwargs={'max_depth': 3, 'branching_factor': 2,}

    response = engine.query(prompt)

    print(f"""
        {disclaimer} \n ---- \n {response} \n ---- \n
        """)

#TODO: funciones para document retrieval con llms usando VectorIndexAutoRetriever
#TODO: actualizar index con nuevos documentos, usando for doc in documents: index.insert(doc) o index.insert_nodes(nodes)
