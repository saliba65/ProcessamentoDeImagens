'''
    Disciplina: Processamento e Análise de Imagens - Ciência da Computação, PUC Minas
    Professor: Alexei 
    Autores: Fernanda Passos, Lucas Saliba e Ygor Melo
'''

import sys
from PyQt6.QtWidgets import QApplication, QWidget

# Declarando um objeto para criar a aplicação 
app = QApplication(sys.argv)

# Declarando um objeto para criação da janela principal
windown = QWidget()
windown.resize(800, 600)
windown.setWindowTitle("Segmentação e Classificação de Imagens Mamográficas - Processamento e Análise de Imagens, PUC Minas")
windown.show()

app.exec()

