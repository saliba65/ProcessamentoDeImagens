'''
    Disciplina: Processamento e Análise de Imagens - Ciência da Computação, PUC Minas
    Professor: Alexei 
    Autores: Fernanda Passos, Lucas Saliba e Ygor Melo
'''

# import PySimpleGUI as sg

# # Layout
# sg.theme('DarkAmber')
# layout = [ 
#             [sg.Text('Segmentação e Classificação de Imagens Mamográficas - Processamento e Análise de Imagens, PUC Minas')],
#             [sg.Button('Carregar imagem', size=(15,1)), 
#             sg.Button('Zoom', size=(15,1)),
#             sg.Button('Contraste', size=(15,1)),
#             sg.Button('Sair', size=(15,1))] ]

# # Window
# window = sg.Window('Processamento e Análise de Imagens', layout, size=(800, 600))

# # Ler eventos
# while True:
#     event, values = window.read()
#     if event == sg.WIN_CLOSED or event == 'Sair': 
#         break
#     if event == 'Carregar imagem': 
#         print('Carregamento de imagem concluido')
#     if event == 'Zoom': 
#         print('Zoom concluido')
#     if event == 'Contraste': 
#         print('Contraste concluido')
        
# window.close()

import sys
from PyQt6.QtWidgets import QApplication, QWidget, QMenu, QMenuBar, QVBoxLayout

''' -------------------------------- INTERFACE --------------------------------'''

# Criando interface utilizando a biblioteca do PyQt

# Declarando um objeto para criar a aplicação 
app = QApplication(sys.argv)

# Declarando um objeto para criação da janela principal
windown = QWidget()
windown.resize(800, 600)
windown.setWindowTitle("Segmentação e Classificação de Imagens Mamográficas - Processamento e Análise de Imagens, PUC Minas")

''' -------------------------------- BARRA DE MENU --------------------------------'''
# Criação da barra de menu
windown.layout = QVBoxLayout()
windown.setLayout(windown.layout)

windown.menuBar = QMenuBar()

# Criação dos botões da barra de menu
windown.fileMenu = QMenu('Abrir Imagem')
windown.menuBar.addMenu(windown.fileMenu)
windown.fileMenu.addAction('Teste', lambda: print('Teste'))
windown.layout.setMenuBar(windown.menuBar)

windown.zoomMenu = QMenu('Zoom')
windown.menuBar.addMenu(windown.zoomMenu)
windown.zoomMenu.addAction('Zoom In', lambda: print('Zoom In'))
windown.zoomMenu.addAction('Zoom Out', lambda: print('Zoom Out'))
windown.layout.setMenuBar(windown.menuBar)

windown.exitMenu = QMenu('Sair')
windown.menuBar.addMenu(windown.exitMenu)
windown.layout.setMenuBar(windown.menuBar)

windown.show()
app.exec()