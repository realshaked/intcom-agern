# Adiciona a raiz do projeto ao sys.path para que os testes importem ag.py e utils.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
