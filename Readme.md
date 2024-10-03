# RAG sobre bases documentales | RAG over Document Databases
----

Este repositorio contiene un sistema de Recuperación y Generación de Respuestas (RAG) diseñado para analizar y resumir documentos mediante el uso de modelos grandes de lenguaje (LLMs).

*This repository contains a Retrieval-Augmented Generation (RAG) system designed to analyze and summarize documents using large language models (LLMs).*

## Actualización principal | Major Update

Se han implementado cambios significativos que mejoran la modularización, eficiencia y enfoque en el multiprocesamiento:

 - Reimplementación del CustomPDFReader: Se ha rehecho desde cero para optimizar la extracción de objetos y gestionar el multiprocesamiento a nivel de página, preservando relaciones estructurales del documento.
 - Rollback del CustomExcelReader: Se ha eliminado temporalmente para evitar demasiadas piezas en movimiento. Será reintroducido en futuras versiones.
 - Nueva clase CustomSimpleDirectoryReader: Basada en SimpleDirectoryReader, esta clase maneja lectores con estrategias multiproceso propias, permitiendo una lectura más eficiente y escalable de lotes de documentos.
 - Adopción de Ollama: Se ha pasado a utilizar Ollama para los LLM, por su capacidad para gestionar de forma óptima la cuantización y la configuración de hiperparámetros bajo el capó.
 - Estrategia de generación de nodos basada en similitud semántica: En lugar de usar tamaños de chunk, los nodos se generan en función de la similitud semántica, lo que permite una mejor coherencia en la representación de la información.
 - Nueva modularización: Se ha mejorado la estructura del código para facilitar su mantenimiento y gestión.
 - Archivo de configuración: Se ha pasado a gestionar la configuración a través de un archivo (config.json), centralizando los ajustes del sistema.

 - *The CustomPDFReader has been completely reworked for better object extraction and handling page-level multiprocessing, while preserving the document’s structural relationships.*
 - *The CustomExcelReader has been temporarily rolled back to avoid too many moving parts. It will be reintroduced in future versions*.
 - *A new class CustomSimpleDirectoryReader has been added, building on SimpleDirectoryReader, to handle readers with their own multiprocessing strategies, allowing for more efficient and scalable reading of document batches.*
 - *Ollama is now adopted, providing better under the hood hyperparameter management and built-in quantization support.*
 - *The node generation strategy now uses semantic similarity instead of chunk sizes, improving coherence in information representation.*
 - *Code modularization has been improved for easier maintenance and management.*
 - *Configuration is now handled through a config.json file, centralizing system settings.*

## Instalación y Configuración | Installation and Configuration

 - **Requisitos | Requirements:** Python v3.8+, conexión a internet (para descargar los modelos). *Internet connection (to download the models)*.

 - **Clona el repositorio | Clone the repository:**

  ```bash
  git clone https://github.com/zimurg/rag_prototype.git
  cd rag_prototype
  ```
 - **Instalación de Ollama | Set up Ollama:** (Opcional | *Optional*)

 ```bash
  apt-get install pciutils
  curl -fsSL https://ollama.com/install.sh
  ```

En pods y entornos containerizados, es necesario fijar la variable global OLLAMA_HOST = "0.0.0.0", y abrir el puerto :11434

*Within pods and containerized environments, it is necessary to set the global vairable OLLAMA_HOST  = "0.0.0.0", and open port :11434*

## Utilización | Usage

El programa se ejecuta mediante el archivo init.py, que permite la carga, resumen y consulta de documentos. Usa los siguientes comandos:

*The program runs through init.py, which allows for loading, summarizing, and querying documents. Use the following commands:*

  ```bash
  python init.py --prompt "tu pregunta aquí / your prompt here"

  ```
### Opciones disponibles | Available options:

    --update: Actualiza las dependencias.
    *Updates libraries*

    --data_dir: Directorio de los documentos (por defecto: Data).
    *Directory of documents (default: Data)*.

    --prompt: La consulta o pregunta a realizar.
    *The query or question to be asked*.

    --files: Archivos específicos a procesar. Permite múltiples archivos.
    *Specific files to process, allowing multiple files*.

    --summarize: Activa el modo resumen de documentos.
    *Enables document summarization*.

## Cambios pendientes y tareas futuras | *Pending Changes and Future Tasks*
#### Implementado | *Implemented*

 - [x] Instalación automática de dependencias al inicio.
 - [x] Modularidad mejorada (refactorización de core.py, init.py y módulos auxiliares).
 - [x] Generación de nodos por similitud semántica.

#### Por implementar | *To be Implemented*

**Preprocesamiento de documentos** | ***Document Preprocessing***

 - [ ] Estrategia para realizar preprocesamiento sólo ante nuevos documentos o cambios significativos (core.py).
 - [ ] Estrategia para manejar la extracción de palabras clave y jerarquías para mejorar el almacenamiento en el índice​ (core.py).
 - [ ] Trasladar parámetros como los tamaños de la ventana de contexto y de la salida LLM al fichero de configuración​ (environment.py).
 - [ ] Extracción y clasificación de imágenes según tipo, para aplicar captioning y OCR cuando sea necesario​. (custom_pdf_reader.py). Introducción del contenido del caption como placeholder en lugar del objeto.

**Multiproceso** | ***Multiprocessing***

 - [ ] Evaluar cuellos de botella en el multiprocesamiento cuando el número de páginas de un documento excede los procesadores disponibles. Mejorar la estrategia de queuing (custom_pdf_reader.py)​.
 - [ ] Extraer las páginas de todos los PDF como si se tratase de un solo archivo, evitando que queden núcleos inactivos (custom_pdf_reader.py).

**Modelos** | ***Models***

 - [ ] Integrar una comprobación periódica para verificar que el servicio de Ollama sigue en funcionamiento (environment.py)​.
 - [ ] Extraer los hiperparámetros de cada modelo de la clase creada por ollama, de forma que puedan configurarse por el usuario (init.py).
 - [ ] Extraer los hiperparámetros del modelo a ficheros de configuración editables por el usuario (models.py)​
 - [ ] Implementar un chequeo para permitir la reescritura del modelo​.

**Permisos y acceso** | ***Permissions and Access***

 - [ ] Crear un sistema de control de permisos, restringiendo acceso a nodos del VectorStore según los permisos del usuario ​​(utils.py, core.py).
 - [ ] Introducir permisos de acceso en las rutas de datos (utils.py)​.

**Funcionalidad de índices y consultas** | **Indexing and Query Functionalities**

 - [ ] Guardar y cargar los VectorStore en disco, evitando que se generen para cada consulta
 - [ ] Implementar funciones para document retrieval con LLMs usando VectorIndexAutoRetriever​ (core.py).
 - [ ] Implementar retrieval de objetos, ej. Gráficas e imágenes (core.py).
 - [ ] Implementar RAG para varios VectorStore (core.py).
 - [ ] Mejorar las estrategias de búsqueda en el motor de consultas para incluir estrategias de traversal, como breadth-first, depth-first, y relevance-guided (query_engine.py)​.

**Varios** | ***Miscellaneous***

 - [ ] Introducir metadatos en los documentos que no los tengan o ampliarlos donde sea necesario​
 - [ ] Implementar lógica para crear un sistema de actualización de índices que no requiera reindexar todo cuando se añadan documentos​
 - [ ] Agregar soporte para más tipos de archivos como .csv y bases de datos SQL​
 - [ ] Limpiar los nombres de archivos con caracteres problemáticos​.
 - [ ] UI para mayor accesibilidad y control del sistema.
 - [ ] Reintroducción de CustomExcelReader con una mejor gestión del multiproceso.
 - [ ] Adaptación al modelo multimodal Pixtral cuando sea totalmente compatible con las librerías.
 - [ ] Reformulación de preguntas para optimización de búsquedas.
 - [ ] Cross-linking en los metadatos entre los nodos y los objetos extraídos

----

**Nota:** Este prototipo no está homologado para su uso en aplicaciones críticas según la normativa vigente de IA. No utilices las respuestas generadas para tomar decisiones sin validarlas con la documentación correspondiente.
