from llama_index.core.readers.file.base import SimpleDirectoryReader
import multiprocessing
import warnings
from functools import reduce
from itertools import repeat
from typing import Any, Dict, List, Optional
from pathlib import Path
from llama_index.core.schema import Document
from tqdm import tqdm
import fsspec

class CustomSimpleDirectoryReader(SimpleDirectoryReader):
    def load_data(
        self,
        show_progress: bool = True,
        num_workers: Optional[int] = None,
        fs: Optional[fsspec.AbstractFileSystem] = None,
        ) -> List[Document]:

        #Cargar los datos del directorio, gestionando los readers que tienen su propia lógica de multiproceso.
        documents = []
        files_to_process = self.input_files
        fs = fs or self.fs

        default_file_reader_cls = SimpleDirectoryReader.supported_suffix_fn()
        default_file_reader_suffix = list(default_file_reader_cls.keys())

        files_with_custom_multiprocessing = []
        other_files = []

        #Agrupar los ficheros dependiendo de si sus lectores utilizan la estrategia de multiproceso por defecto o no
        for input_file in files_to_process:
            file_suffix = input_file.suffix.lower()
            if file_suffix in default_file_reader_suffix or file_suffix in self.file_extractor:
                # Use file readers
                if file_suffix not in self.file_extractor:
                    reader_cls = default_file_reader_cls[file_suffix]
                    self.file_extractor[file_suffix] = reader_cls()
                reader = self.file_extractor[file_suffix]
                if getattr(reader, 'handles_multiprocessing', False):
                    files_with_custom_multiprocessing.append(input_file)
                else:
                    other_files.append(input_file)
            else:
                other_files.append(input_file)

        # Procesar los ficheros con lógica propia de multiproceso de forma secuencial
        for input_file in files_with_custom_multiprocessing:
            docs = SimpleDirectoryReader.load_file(
                input_file=input_file,
                file_metadata=self.file_metadata,
                file_extractor=self.file_extractor,
                filename_as_id=self.filename_as_id,
                encoding=self.encoding,
                errors=self.errors,
                raise_on_error=self.raise_on_error,
                fs=fs,
            )
            documents.extend(docs)

        #Usar la estrategia por defecto para los demás ficheros
        if num_workers and num_workers > 1:
            if num_workers > multiprocessing.cpu_count():
                warnings.warn(
                    "Specified num_workers exceed number of CPUs in the system. "
                    "Setting `num_workers` down to the maximum CPU count."
                )
            with multiprocessing.get_context("spawn").Pool(num_workers) as p:
                results = p.starmap(
                    SimpleDirectoryReader.load_file,
                    zip(
                        other_files,
                        repeat(self.file_metadata),
                        repeat(self.file_extractor),
                        repeat(self.filename_as_id),
                        repeat(self.encoding),
                        repeat(self.errors),
                        repeat(self.raise_on_error),
                        repeat(fs),
                    ),
                )
                documents.extend(reduce(lambda x, y: x + y, results))
        else:
            if show_progress:
                other_files = tqdm(other_files, desc="Loading files", unit="file")
            for input_file in other_files:
                docs = SimpleDirectoryReader.load_file(
                    input_file=input_file,
                    file_metadata=self.file_metadata,
                    file_extractor=self.file_extractor,
                    filename_as_id=self.filename_as_id,
                    encoding=self.encoding,
                    errors=self.errors,
                    raise_on_error=self.raise_on_error,
                    fs=fs,
                )
                documents.extend(docs)

        return self._exclude_metadata(documents)
