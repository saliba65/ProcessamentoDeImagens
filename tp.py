'''
    Disciplina: Processamento e Análise de Imagens - Ciência da Computação, PUC Minas
    Professor: Alexei 
    Autores: Lucas Saliba (650625) e Ygor Melo
'''

import tkinter 
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image 

# Usando tkinter para leitura de imagens
#Permitindo buscar arquivos png e tiff na biblioteca 
root = tkinter.Tk() 
filetypes = (('Png files', '*.png'),('Tiff files', '*.tiff'))
filename = askopenfilename(filetypes=filetypes)

img = ImageTk.PhotoImage(Image.open(filename)) 
panel = tkinter.Label(root, image = img) 
panel.pack(side = "bottom", fill = "both", 
           expand = "yes") 

root.mainloop()