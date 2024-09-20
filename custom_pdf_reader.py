#custom_pdf_reader.py

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
import pdfplumber
from tabulate import tabulate
import os

class CustomPDFReader(BaseReader):
    def load_data(self, file, **kwargs):
        file_path = str(file)
        extra_info = kwargs.get('extra_info', {})
        documents = []
        try:
            text = ''
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extraer texto de la página
                    page_text = page.extract_text()
                    if page_text:
                        text += f'\n--- Página {page_num + 1} ---\n{page_text}'
                    else:
                        text += f'\n--- Página {page_num + 1} ---\n[No se pudo extraer texto de esta página]'

                    # Extraer tablas de la página
                    tables = page.extract_tables()
                    for table_num, table in enumerate(tables):
                        if table:
                            markdown_table = tabulate(table, tablefmt="pipe")
                            text += f'\n--- Tabla {table_num + 1} en la Página {page_num + 1} ---\n{markdown_table}'

                    # TODO: extracción de imágenes (para captioning y RAG)
                    # images = page.images
                    # for image_num, image in enumerate(images):
                    #     # Procesar las imágenes según tus necesidades
                    #     pass

            documents.append(Document(text=text, doc_id=os.path.basename(file_path), extra_info=extra_info))
        except Exception as e:
            print(f"Error al leer {file_path}: {e}")
            raise e

        return documents
