# excel_reader.py

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
import pandas as pd
import os

class ExcelReader(BaseReader):
    def load_data(self, file, **kwargs):
        file = str(file)
        extra_info = kwargs.get('extra_info', {})
        documents = []
        try:

            #pd.set_option('compute.use_bottleneck', False) TODO: eliminar si no se detectan problemas de incorrecta asignación de hilos de la cpu

            # Leer todas las hojas del archivo Excel
            dfs = pd.read_excel(file, sheet_name=None)
            text = ''
            # Iterar sobre cada hoja y su DataFrame
            for sheet_name, df in dfs.items():
                # Agregar el nombre de la hoja
                text += f'\n--- Hoja: {sheet_name} ---\n'
                # Convertir el DataFrame a cadena y agregarlo al texto
                text += df.to_string(index=False)
            # Crear un objeto Document con el texto y el ID del documento
            documents.append(Document(text=text, doc_id=os.path.basename(file), extra_info=extra_info))
        except Exception as e:
            print(f"Error al leer {file}: {e}")
        return documents
