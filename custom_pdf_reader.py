# pdf_reader.py

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
import camelot
import fitz  # PyMuPDF
import os

class CustomPDFReader(BaseReader):
    def load_data(self, file: str):
        # Lista para almacenar los documentos procesados
        documents = []
        try:
            # Abrir el archivo PDF usando PyMuPDF
            doc = fitz.open(file)
            num_pages = doc.page_count
            text = ''
            # Iterar sobre cada página del PDF
            for page_num in range(num_pages):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                # Agregar el texto de la página al contenido
                text += f'\n--- Página {page_num + 1} ---\n{page_text}'

                # Extraer tablas usando Camelot en la página actual
                tables = camelot.read_pdf(file, pages=str(page_num + 1))
                for table in tables:
                    # Convertir la tabla a formato Markdown para preservar la estructura
                    markdown_table = table.df.to_markdown(index=False)
                    # Agregar la tabla en formato Markdown al contenido
                    text += f'\n--- Tabla en la Página {page_num + 1} ---\n{markdown_table}'

            doc.close()
            # Crear un objeto Document con el texto completo y el ID del documento
            documents.append(Document(text=text, doc_id=os.path.basename(file)))
        except Exception as e:
            print(f"Error al leer {file}: {e}")
        return documents
