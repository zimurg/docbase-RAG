# custom_pdf_reader.py

import pdfplumber
import multiprocessing as mp
from typing import List, Dict, Optional
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from functools import partial
import logging
import os
import re

logging.basicConfig(filename='pdf_processing_incidents.log', level=logging.WARN) #TODO: incrustar los resultados de logging dentro de los ficheros, de forma que durante la recuperación se puedan identificar los elementos perdidos.

def safe_str(item):
    return str(item) if item is not None else ""

class CustomPDFReader(BaseReader):
    """
    Reader customizado para extraer texto, tablas y metadatos de ficheros PDF, preservando sus relaciones estructurales, y utilizando multiporocesameiento a nivel de página.
    Su salida es texto con un formato basado en HTML con limitadores explícitos y metadatos incrustados.
    Diseñado para funcionar junto a LlamaIndex, aunque puede utilizarse de forma independiente.

    En desarrollos futuros se integrará con un módulo para clasificar y generar captions para los objetos incrustados, preparando las salidas para estrategias de búsqueda y recuperación por relaciones a través de crossreferencia. También se implementarán funciones de OCR.
    """

    handles_multiprocessing = True
    #Al usar lectores de directorios, es necesario usar un módulo personalizado para, que identifique el atributo handles_multiprocessing y adapte su estrategia de multiproceso para evitar conflictos con el spawning de procesos

    def __init__(
        self,
        use_multiprocessing: bool = True,
        skip_headers_and_footers: bool = True,
        caption_non_text_objects: bool = False,
        captioning_module: Optional[object] = None
    ):

        self.use_multiprocessing = use_multiprocessing
        self.skip_headers_and_footers = skip_headers_and_footers
        self.caption_non_text_objects = caption_non_text_objects
        self.captioning_module = captioning_module #TODO: módulo de captioning
        self.detected_headers = set()
        self.detected_footers = set()

    def load_data(self, file_path: str, extra_info: dict = {}, **kwargs) -> List[Document]:

        try:
            with pdfplumber.open(file_path) as pdf:
                metadata = pdf.metadata
                total_pages = len(pdf.pages)
        except Exception as e:
            logging.error(f"Error opening PDF file {file_path}: {str(e)}")
            return []

        self.total_pages = total_pages

        metadata = self._extract_metadata(metadata)
        if extra_info:
            metadata.update({key: safe_str(value) for key, value in extra_info.items()})

        html_parts = [self._generate_html_front_matter(metadata), "<document_start>\n\n"]

        # Reglas para decidir si usar multiporocesameiento
        is_main_process = (mp.current_process().name == 'MainProcess')
        use_mp = self.use_multiprocessing and is_main_process and total_pages > 1
        #TODO: estudiar los problemas y cuellos de botella que surgen cuando se procesan documentos con más páginas que los procesadores disponibles. Evaluar si es conveniente una mejor estrategia de queuing

        page_numbers = list(range(total_pages))

        header_candidates = []
        footer_candidates = []
        page_html_results = []

        if use_mp:
            num_processes = min(mp.cpu_count(), total_pages)
            with mp.Pool(processes=num_processes) as pool:
                func = partial(self._process_single_page, file_path)
                results = pool.map(func, page_numbers)
            for page_html, headers, footers in results:
                page_html_results.append(page_html)
                header_candidates.extend(headers)
                footer_candidates.extend(footers)
        else:
            # Procesamiento secuencial
            for page_num in page_numbers:
                page_html, headers, footers = self._process_single_page(file_path, page_num)
                page_html_results.append(page_html)
                header_candidates.extend(headers)
                footer_candidates.extend(footers)

        # Determinar y eliminar encabezados y pies de página
        self._determine_headers_and_footers(header_candidates, footer_candidates)

        final_html_parts = []
        for page_html in page_html_results:
            cleaned_page_html = self._remove_detected_headers_and_footers(page_html)
            final_html_parts.append(cleaned_page_html)

        html_parts.extend(final_html_parts)
        html_parts.append("<document_end>\n\n")
        html_content = ''.join(html_parts)

        # Crear el objeto Documento
        document = Document(
            text=html_content,
            metadata={
                "file_path": file_path,
                "total_pages": total_pages,
                **metadata,
            }
        )

        return [document]

    def _process_single_page(self, file_path: str, page_num: int) -> ([str, List[str], List[str]]):
        try:
            with pdfplumber.open(file_path) as pdf:
                if page_num >= len(pdf.pages):
                    return (f"<!-- Page {page_num + 1} out of range -->\n", [], [])
                page = pdf.pages[page_num]
                page_html, headers, footers = self._process_single_page_content(page, page_num + 1)
            return (page_html, headers, footers)
        except Exception as e:
            logging.error(f"Error processing page {page_num + 1}: {str(e)}")
            return (f"<!-- Error processing page {page_num + 1} -->\n", [], [])


    def _process_single_page_content(self, page, page_num: int) -> ([str, List[str], List[str]]):

        html_parts = [f"\n\n<!-- Page {page_num} -->\n\n"]

        tables_html = self._extract_tables(page)

        text_html, headers, footers = self._extract_text_blocks(page)

        images_html = ""
        if self.caption_non_text_objects and self.captioning_module is not None:
            images_html = self._extract_and_caption_images(page, page_num)

        html_parts.extend([text_html, tables_html, images_html])
        page_html = ''.join(html_parts)
        return (page_html, headers, footers)


    def _extract_tables(self, page) -> str:

        tables_html_parts = []
        extracted_tables = page.extract_tables()
        for table_id, table in enumerate(extracted_tables):
            table_html = self._convert_table_to_html(table, table_id)
            if table_html:
                tables_html_parts.append(table_html)
        return ''.join(tables_html_parts)


    def _extract_text_blocks(self, page) -> ([str, List[str], List[str]]):

        text_html_parts = []
        headers = []
        footers = []

        try:
            words = page.extract_words(use_text_flow=True, keep_blank_chars=True)
            if not words:
                return ("", [], [])

            lines = self._group_blocks_into_lines(words)

            page_height = page.height
            threshold = page_height * 0.1  # Ajustar si es necesario.

            for line in lines:
                line_text = ' '.join([word['text'] for word in line])
                y0 = min(word['top'] for word in line)
                y1 = max(word['bottom'] for word in line)

                if y0 < threshold:
                    headers.append(line_text)
                    # Se incluye el tag en el contenido, indicando candidatos para eliminación
                    text_html_parts.append(f"<header_candidate>{line_text}</header_candidate>\n")

                elif y1 > (page_height - threshold):
                    footers.append(line_text)
                    text_html_parts.append(f"<footer_candidate>{line_text}</footer_candidate>\n")

                else:
                    line_html = self._process_text_line(line_text)
                    if line_html:
                        text_html_parts.append(line_html)

        except Exception as e:
            logging.error(f"Error extracting text from page: {str(e)}")

        return (''.join(text_html_parts), headers, footers)


    def _group_blocks_into_lines(self, blocks):

        lines = []
        blocks_sorted = sorted(blocks, key=lambda x: (x['top'], x['x0']))
        current_line = []
        current_top = None

        for block in blocks_sorted:
            if current_top is None or abs(block['top'] - current_top) < 2:  # Umbral para la misma línea
                current_line.append(block)
                current_top = block['top']

            else:
                lines.append(current_line)
                current_line = [block]
                current_top = block['top']

        if current_line:
            lines.append(current_line)

        return lines


    def _determine_headers_and_footers(self, header_candidates: List[str], footer_candidates: List[str]):

        header_counts = {}
        footer_counts = {}
        total_pages = self.total_pages

        for text in header_candidates:
            header_counts[text] = header_counts.get(text, 0) + 1

        for text in footer_candidates:
            footer_counts[text] = footer_counts.get(text, 0) + 1

        header_footer_cutoff = 0.75 * total_pages  # 75% threshold

        self.detected_headers = {
            text for text, count in header_counts.items()
            if count >= header_footer_cutoff
        }

        self.detected_footers = {
            text for text, count in footer_counts.items()
            if count >= header_footer_cutoff
        }


    def _is_page_number_footer(self, text: str) -> bool:
        #Evalúa si el texto coincide con patrones que identifican números de página, para su eliminación.

        text = text.strip()
        patterns = [
            r'^[Pp]age\s*\d+?$',   # Detecta 'page 1', 'Page 2', etc.
            r'^[Pp].gina\s*\d+?$', # Detecta 'pagina 1', 'Página 2', etc.
            r'^[Pp]\.\s*\d+?$',    # Dtecta 'p.1', 'P. 2', etc.
            r'^\d+?$',             # Detecta '1', '2', etc.
        ]
        for pattern in patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True

        return False

    def _remove_page_number_footers(self, page_html: str) -> str:

        pattern = r'<footer_candidate>(.*?)</footer_candidate>\n'
        matches = re.findall(pattern, page_html, flags=re.IGNORECASE)
        for match in matches:
            if self._is_page_number_footer(match):
                full_tag = f"<footer_candidate>{match}</footer_candidate>\n"
                page_html = page_html.replace(full_tag, '')

        return page_html


    def _remove_detected_headers_and_footers(self, page_html: str) -> str:

        for header in self.detected_headers:
            pattern = re.escape(f"<header_candidate>{header}</header_candidate>\n")
            page_html = re.sub(pattern, '', page_html, flags=re.IGNORECASE)

        for footer in self.detected_footers:
            pattern = re.escape(f"<footer_candidate>{footer}</footer_candidate>\n")
            page_html = re.sub(pattern, '', page_html, flags=re.IGNORECASE)

        page_html = self._remove_page_number_footers(page_html)

        # Retira los tags de los candidatos a encabezados y pies de página que no han sido eliminados
        page_html = re.sub(r'</?header_candidate>', '', page_html)
        page_html = re.sub(r'</?footer_candidate>', '', page_html)

        return page_html


    def _process_text_line(self, line: str) -> str:

        stripped_line = line.strip()
        if not stripped_line:
            return ""

        # Detecta encabezados y elementos especiales
        if stripped_line.isupper():
            return f"<header>{stripped_line}</header>\n\n"
        elif stripped_line.startswith(('1.', '2.', '3.', '-', '*', '•')):
            content = stripped_line.lstrip('1234567890.-*• ').strip()
            return f"<list-item>{content}</list-item>\n"
        else:
            if self._is_figure_reference(stripped_line):
                figure_desc = self._extract_figure_description(stripped_line)
                return f"<figure>{figure_desc}</figure>\n\n"
            else:
                return f"<paragraph>{stripped_line}</paragraph>\n\n"

    def _extract_and_caption_images(self, page, page_num: int) -> str:
        #Extrae las imágenes de la página, opcionalmente generando sus captions.

        images_html_parts = []
        try:
            images = page.images
            if not images:
                return ""
            for img_index, img in enumerate(images):

                image_obj = page.extract_image(img["object_id"])
                image_data = image_obj["image"]
                image_ext = image_obj["ext"]
                image_name = f"page{page_num}_img{img_index}.{image_ext}"

                # TODO: Guardar la imagen por separado, manteniendo referencias cruzadas
                # TODO: Clasificar las imágenes según el tipo, tal vez con una NN pequeña, de forma que podamos distinguir grafos, gráficas, texto escaneado (para OCR), figuras (no se procesarán), fotografías, etc, y procesarlas del modo más adecuado.

                image_category = "graph"  # Categoría placeholder

                # Se genera el caption con el módulo correspondiente
                if self.captioning_module:
                    caption = self.captioning_module.generate_caption(image_data)
                else:
                    caption = "Una imagen"

                images_html_parts.append(f"<image src='{image_name}' category='{image_category}' page='{page_num}'>\n")
                images_html_parts.append(f"<caption>{caption}</caption>\n")
                images_html_parts.append("</image>\n\n")

        except Exception as e:
            logging.error(f"Error al extraer imágenes en la página {page_num}: {str(e)}")

        return ''.join(images_html_parts)


    def _generate_html_front_matter(self, metadata: Dict[str, str]) -> str:

        front_matter = "<metadata>\n"
        for key, value in metadata.items():
            front_matter += f"  <{key}>{safe_str(value)}</{key}>\n"
        front_matter += "</metadata>\n\n"

        return front_matter


    def _extract_metadata(self, metadata: Dict[str, str]) -> Dict[str, str]:

        cleaned_metadata = {k.replace(" ", "_").lower(): safe_str(v) for k, v in metadata.items()}

        return cleaned_metadata


    def _convert_table_to_html(self, table: List[List[str]], table_id: int) -> str:

        if not table or not any(table):
            return ""

        html_parts = [f"<table id='table_{table_id}'>\n"]
        header = table[0]

        html_parts.append("  <tr>\n")
        for cell in header:
            html_parts.append(f"    <th>{safe_str(cell)}</th>\n")
        html_parts.append("  </tr>\n")

        for row in table[1:]:
            html_parts.append("  <tr>\n")
            for cell in row:
                html_parts.append(f"    <td>{safe_str(cell)}</td>\n")
            html_parts.append("  </tr>\n")

        html_parts.append("</table>\n")

        return ''.join(html_parts)


    def _is_figure_reference(self, line: str) -> bool:

        return line.startswith((r'Fig?', 'Figura', 'Figure'))


    def _extract_figure_description(self, line: str) -> str:

        parts = line.split(':', 1)
        if len(parts) == 2:
            return parts[1].strip()

        else:

            parts = line.split('.', 1)
            if len(parts) == 2:
                return parts[1].strip()

            else:
                return line.strip()
