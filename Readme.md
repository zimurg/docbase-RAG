# RAG sobre bases documentales

Este repositorio contiene un prototipo de un sistema de Recuperación y Generación de Respuestas (RAG) diseñado para analizar y resumir documentos de manera automatizada utilizando grandes modelos de lenguaje.

Actualmente, se emplean los modelos Mistral 7B y Nomic

## Requisitos

Antes de ejecutar el programa, es necesario cumplir con los siguientes requisitos:

- **API Key de Huggingface**: Debes configurar tu clave de API de Huggingface en la variable de entorno `HF_TOKEN`. Además, necesitas obtener acceso a los modelos que se utilizarán desde la página web de Huggingface.
- **Python**: Versión 3.8 o superior.
- **Dependencias**: Las dependencias necesarias se encuentran listadas en el archivo `requirements.txt`. Se recomienda instalar todas las dependencias con el siguiente comando:


  ```bash
  pip install -r requirements.txt
  ```

## Instalación y Configuración

Clona el repositorio:

   ```bash
   git clone https://github.com/tu_usuario/rag_prototype.git
   cd rag_prototype
   ```

Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

Configura la variable de entorno HF_TOKEN con tu clave de API de Huggingface:

   ```bash
   export HF_TOKEN={tu_token_de_huggingface}
   ```

## Utilzación

El programa se ejecuta desde archivo main.py, que permite la carga, resumen y consulta de documentos de diversos formatos (PDF, DOCX, Markdown, etc.).

#### Comandos básicos

### Ejecutar el programa:

   ```bash
   python main.py --prompt "tu pregunta aquí"
   ```

### Opciones disponibles:

    --data_dir: Directorio que contiene los documentos (por defecto: Data).
    --prompt: La consulta o pregunta que deseas realizar al modelo.
    --file: Especifica un archivo en particular (relativo a data_dir).
    --summarize: Activa el resumen de documentos.
    --no_compliance_eval: Desactiva la evaluación de cumplimiento.

## Desarrollos futuros

 - [ ] Instalación automática de las dependencias al ejecutar el programa
 - [ ] Los documentos solo se cargan la primera vez o cuando se especifica --update_docbase=True.
 - [ ] Permitir la selección de modelos de lenguaje y embeddings y sus hiperparámetros
 - [ ] Asignación de metadatos: En la primera pasada, el modelo asigna metadatos útiles a cada documento según un formato definido en un archivo .json.
 - [ ] Eliminación de caracteres problemáticos: Se eliminan nombres de archivo con caracteres problemáticos en la pasada inicial.
 - [ ] Reformulación de preguntas y estrategia agencial: Permite al modelo reformular preguntas y optimizar las búsquedas en la base documental.
 - [ ] OCR y captioning de imágenes: Extracción y etiquetado de imágenes en documentos durante la fase inicial. Inclusión de captions en metadatos
 - [ ] Control de permisos: Restringir el acceso a los documentos según los permisos del usuario, integrando una lista de permisos del gestor documental con los metadatos.
 - [ ] Verbose Mode: --verbose=False (por defecto) elimina todos los warnings no necesarios.
 - [ ] Gestión de dependencias: Instalación y actualización de dependencias integradas al inicio del programa.
 - [ ] Modularidad y refactorización: Separación de módulos principales (main.py, init.py y módulos auxiliares) para mejorar la estructura del código.
 - [ ] Adaptación a cambios en llamaindex: Ajustar los imports a las últimas actualizaciones del proyecto llama_hub.

**Nota:** Este prototipo no está homologado para su uso en aplicaciones críticas según la normativa vigente de IA. No utilices las respuestas generadas para tomar decisiones sin validarlas con la documentación correspondiente.
