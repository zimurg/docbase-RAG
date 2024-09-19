# metadata_parser.py

import re
import json

class MetadataParser:
    def __init__(self, config_file='metadata_config.json'):
        # Cargar la configuración de metadatos desde el archivo JSON
        with open(config_file, 'r') as f:
            config = json.load(f)
        self.metadata_fields = config.get('metadata_fields', [])

    def extract_metadata(self, text):
        # Extraer metadatos del texto usando las expresiones regulares definidas
        metadata = {}
        for field in self.metadata_fields:
            name = field.get('name')
            pattern = field.get('pattern')
            if name and pattern:
                match = re.search(pattern, text)
                if match:
                    metadata[name] = match.group(1)
        return metadata
