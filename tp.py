'''
    Disciplina: Processamento e Análise de Imagens - Ciência da Computação, PUC Minas
    Professor: Alexei 
    Autores: Lucas Saliba (650625) e Ygor Melo
'''

import cv2
import PIL
import PIL.ImageTk
import tkinter
import tkinter.filedialog
import imutils

# Funcao contraste da imagem
def scale_event_1(event: str) -> None:
    image_cv2 = cv2.imread(file_name)

    image_cv2 = cv2.addWeighted(image_cv2, scale_1.get(), 0, 0, 0)

    image_cv2 = cv2.resize(image_cv2, (384, 384))
    
    image = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(image_cv2))

    label_1.configure(image=image)

    label_1.image = image

# Funcao segmentacao da imagem
def scale_event_2(event: str) -> None:
    image_cv2 = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

    # Aplicando limiarizacao binaria
    image_cv2 = cv2.threshold(image_cv2, scale_2.get(), 255, cv2.THRESH_BINARY)[1]

    image_cv2 = cv2.resize(image_cv2, (384, 384))
    
    image = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(image_cv2))

    label_1.configure(image=image)

    label_1.image = image

# Funcao zoom na imagem
def scale_event_3(event: str) -> None:
    image_cv2 = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

    image_cv2 = cv2.resize(image_cv2, None, fx=scale_3.get() / 10 + 1, fy=scale_3.get() / 10 + 1) 
    
    image = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(image_cv2))

    label_1.configure(image=image)

    label_1.image = image

# Usando tkinter para leitura de imagens
#Permitindo buscar arquivos png e tiff na biblioteca 

root = tkinter.Tk()
root.geometry('512x512')
filetypes = (('PNG File', '*.png'), ('TIFF File', '*.tiff'))
file_name = tkinter.filedialog.askopenfilename(filetypes=filetypes)

# Redimencionando a imagem para o tamanho ideal da janela
image = PIL.Image.open(file_name)
image = image.resize((384, 384))

image = PIL.ImageTk.PhotoImage(image)

# Criacao do slider para alteracao de contraste
scale_1 = tkinter.Scale(
    root,
    command=scale_event_1,
    orient='horizontal',
    label='CONTRASTE',
    from_=0,
    to=255
)

# Criacao do slider de segmentacao
scale_2 = tkinter.Scale(
    root,
    command=scale_event_2,
    orient='horizontal',
    label='SEGMENTAÇÃO',
    from_=0,
    to=255
)

# Zoom image
scale_3 = tkinter.Scale(
    root,
    command=scale_event_3,
    orient='horizontal',
    label='ZOOM',
    from_=0,
    to=7
)

#Chamada de componentes
label_1 = tkinter.Label(root, image=image)

scale_1.pack()
scale_2.pack()
scale_3.pack()

label_1.pack()

root.mainloop()
