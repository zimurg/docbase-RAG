# utils.py

import os
import multiprocessing
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import DocxReader, FlatReader, MarkdownReader
from custom_pdf_reader import CustomPDFReader
from excel_reader import ExcelReader
from metadata_parser import MetadataParser

def get_data_path(data_dir="Data"):
    wd = os.getcwd()
    data_path = os.path.join(wd, data_dir)
    return data_path

def update_docs(data_dir="Data", files=None):
    data_path = get_data_path(data_dir=data_dir)
    workers = multiprocessing.cpu_count()

    # Inicializar los lectores de archivos
    pdf_reader = CustomPDFReader()
    docx_reader = DocxReader()
    txt_reader = FlatReader()
    md_reader = MarkdownReader()
    excel_reader = ExcelReader()

    # Diccionario de extractores de archivos
    file_extractor_dict = {
        ".pdf": pdf_reader,
        ".docx": docx_reader,
        ".doc": docx_reader,
        ".txt": txt_reader,
        ".md": md_reader,
        ".xlsx": excel_reader,
        ".xls": excel_reader,
        # TODO: agregar más tipos de archivos, como .csv; agregar compatibilidad con SQL
    }

    metadata_parser = MetadataParser()

    try:
        if files:
            # Procesar archivos específicos
            input_files = [os.path.join(data_path, f) for f in files]
            documents = SimpleDirectoryReader(
                input_files=input_files,
                file_extractor=file_extractor_dict
            ).load_data(num_workers=workers)
        else:
            # Procesar todo el directorio de datos
            documents = SimpleDirectoryReader(
                input_dir=data_path,
                recursive=True,
                file_extractor=file_extractor_dict
            ).load_data(num_workers=workers)

        # Extraer metadatos y agregarlos a los documentos
        for doc in documents:
            metadata = metadata_parser.extract_metadata(doc.text)
            doc.extra_info = metadata  # Guardamos los metadatos en extra_info

    except Exception as e:
        print(f"Ocurrió un error al cargar los documentos: {e}")
        documents = []

    return documents
