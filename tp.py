'''
    Disciplina: Processamento e Análise de Imagens - Ciência da Computação, PUC Minas
    Professor: Alexei 
    Autores: Lucas Saliba (650625) e Ygor Melo
'''

import tkinter
import cv2
import numpy as np
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image 

# Usando tkinter para leitura de imagens
#Permitindo buscar arquivos png e tiff na biblioteca 
root = tkinter.Tk() 
root.geometry("512x512")
filetypes = (('Png files', '*.png'),('Tiff files', '*.tiff'))
filename = askopenfilename(filetypes=filetypes)
# Redimencionando a imagem para o tamanho ideal da janela
img = Image.open(filename)
img = img.resize((384,384))

img = ImageTk.PhotoImage(img) 
panel = tkinter.Label(root, image = img) 
panel.pack(side = "bottom", fill = "both", 
           expand = "yes") 

#Criacao do slider para alteracao de contraste
def slider_changed(event):  
    imgCv2 = cv2.imread(filename)
    out = cv2.addWeighted(imgCv2, slider.get(), imgCv2, 0, 0)
    out = cv2.resize(out, (384,384))
    img = ImageTk.PhotoImage(Image.fromarray(out)) 
    panel.configure(image=img)
    panel.image = img

# Valores do slider - 1 a 256
slider = tkinter.Scale(
    root,
    from_=1,
    to=256,
    orient='horizontal',
    command=slider_changed
) 

slider.pack()


root.mainloop()
