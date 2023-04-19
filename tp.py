'''
    Disciplina: Processamento e Análise de Imagens - Ciência da Computação, PUC Minas
    Professor: Alexei 
    Autores: Fernanda Passos, Lucas Saliba e Ygor Melo
'''

import PySimpleGUI as sg

# Layout
sg.theme('DarkAmber')
layout = [ 
            [sg.Text('Segmentação e Classificação de Imagens Mamográficas - Processamento e Análise de Imagens, PUC Minas')],
            [sg.Button('Carregar imagem', size=(15,1)), 
            sg.Button('Zoom', size=(15,1)),
            sg.Button('Contraste', size=(15,1)),
            sg.Button('Sair', size=(15,1))] ]

# Window
window = sg.Window('Processamento e Análise de Imagens', layout, size=(800, 600))

# Ler eventos
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Sair': 
        break
    if event == 'Carregar imagem': 
        print('Carregamento de imagem concluido')
    if event == 'Zoom': 
        print('Zoom concluido')
    if event == 'Contraste': 
        print('Contraste concluido')
        
window.close()