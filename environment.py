#environment.py

import os, subprocess, sys, shutil, platform
from pathlib import Path
import json

def check_and_install_dependencies():
    marker_file = Path(".dependencies_installed")

    if not marker_file.exists():
        print("Instalando las dependencias...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            marker_file.touch()
            print("Se han instalado las dependencias")
        except subprocess.CalledProcessError as e:
            print(f"Se ha producido un error al instalar las dependencias: {e}")
            sys.exit(1)
    else:
        print("Todas las dependencias se encuentran instaladas.")


def update_dependencies():
    try:
        print("Actualizando las dependencias...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "-r", "requirements.txt"])
        print("Se han actualizado las dependencias.")
    except subprocess.CalledProcessError as e:
        print(f"Se ha producido un error al actualizar las dependencias: {e}")
        sys.exit(1)
    marker_file = Path(".requirements_installed")
    marker_file.touch()


def check_and_install_ollama():

    if shutil.which('ollama') is None:
        print("Ollama no está instalado. Instalando ahora...")
        subprocess.run("(curl -fsSL https://ollama.com/install.sh | sh && ollama serve > ollama.log 2>&1) &", shell=True)
    else:
        print("Ollama ya está instalado.")

    if shutil.which('lspci') is None:
        if platform.system() == 'Linux':
            try:
                import distro
            except ImportError:
                print("El módulo 'distro' no está instalado. Instalándolo ahora...")
                subprocess.run([sys.executable, "-m", "pip", "install", "distro"])
                import distro
            distro_id = distro.id().lower()
            if distro_id in ['debian', 'ubuntu', 'mint']:
                print("Instalando pciutils en un sistema basado en Debian.")
                subprocess.run("apt-get update && apt-get install -y pciutils", shell=True)
            elif distro_id in ['centos', 'redhat', 'fedora']:
                print("Instalando pciutils en un sistema basado en RHEL.")
                subprocess.run("yum install -y pciutils", shell=True)
            else:
                print("Distribución Linux no soportada para instalar pciutils.")
        else:
            print("La instalación de pciutils solo se soporta en Linux.")
    else:
        print("pciutils ya está instalado.")




#TODO: Introducir un check periódico de que ollama sigue en funcionamiento.


def setup_environment(llm, embeddings_model):
    from llama_index.core import Settings

    Settings.llm = llm
    Settings.embed_model = embeddings_model
    Settings.context_window =25000 #TODO: extraer del fichero de configuración del modelo
    Settings.num_output = 512 #TODO: pasar a un arg de inicialización



