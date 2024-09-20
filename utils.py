# utils.py

import os
import multiprocessing
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import (DocxReader,MarkdownReader,FlatReader)
# TODO:desde llama_index>0.6.33, los lectores dependen del módulo llama_hub. Dado que es una iniciativa comunitaria, verificar si cumplen con los requisitos de seguridad. Si existen consideraciones, hacer rollback.
from custom_pdf_reader import CustomPDFReader
#from excel_reader import ExcelReader


def get_data_path(data_dir="Data"):
    wd = os.getcwd()
    data_path = os.path.join(wd, data_dir)
    return data_path

def update_docs(data_dir="Data", files=None):
    data_path = get_data_path(data_dir=data_dir)

    # Establecer variables de entorno para limitar los hilos utilizados por OpenBLAS y MKL para evitar conflictos
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    workers = multiprocessing.cpu_count()

    # Inicializar los lectores de archivos
    pdf_reader = CustomPDFReader()
    docx_reader = DocxReader()
    txt_reader = FlatReader()
    md_reader = MarkdownReader()
    #excel_reader = ExcelReader()

    # Diccionario de extractores de archivos
    file_extractor_dict = {
        ".pdf": pdf_reader,
        ".docx": docx_reader,
        ".doc": docx_reader,
        ".txt": txt_reader,
        ".md": md_reader,
        #".xlsx": excel_reader,
        #".xls": excel_reader,
        # TODO: agregar más tipos de archivos, como .csv; agregar compatibilidad con bases SQL
    }


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
            ).load_data(num_workers=workers) #TODO: Ver como paralelizar, dado que módulos como SmartPDFLoader no son serializables


    except Exception as e:
        print(f"Ocurrió un error al cargar los documentos: {e}")
        documents = []
        raise e

    return documents
