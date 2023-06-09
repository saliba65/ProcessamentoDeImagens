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
import os
import tqdm
import numpy as np 

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

# Verificar se a imagem pertence ao conjunto de testes
def is_test(file: str) -> bool:
    id = int(file.split('(')[1].strip(').png'))

    answer = False
    
    if id % 4 == 0:
        answer = True

    return answer

# Ler diretorios de mamografias separando treino e teste já realizando a conversao para numpy array
def read_data() -> tuple:
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    y_dict = {
        'D': 0,
        'E': 1,
        'F': 2,
        'G': 3
    }

    for folder in tqdm.tqdm(os.listdir('./mamografias')):
            # Condicional para nao lida arquivos MACOSX
            if folder != '__MACOSX':
                for file in os.listdir(f'./mamografias/{folder}'):
                    if file.endswith('.png'):
                        image = PIL.Image.open(f'./mamografias/{folder}/{file}')

                        #Atribuir a conjunto de testes
                        if is_test(file):
                            X_test.append(np.asarray(image))
                            y_test.append(y_dict[folder[0]])
                        #Atribuir a conjunto de treino
                        else:
                            X_train.append(np.asarray(image))
                            y_train.append(y_dict[folder[0]])

    # Conversao para numpy array
    # x = matrizes de pixels
    X_train = np.array(X_train, dtype=object)
    X_test = np.array(X_test, dtype=object)
    # y = birads respectivos 
    y_train = np.array(y_train, dtype=object)
    y_test = np.array(y_test, dtype=object)

    return (X_train, X_test, y_train, y_test)

# Segmentacao automatica de imagens do diretorio
def apply_threshold(X_array: np.array) -> np.array:
    X_transformed = []

    for X in X_array:
        # Aplicacao de segmentacao binaria em cada item recebido
        X_transformed.append(cv2.threshold(X, 7, 255, cv2.THRESH_BINARY)[1])

    return np.array(X_transformed, dtype=object)

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

#Botoes
# button = tkinter.Button(root, text="Ler diretorio", command=read_data(X_train,X_test,y_train,y_test))
# button = tkinter.Button(root, text="Segmentar Treino", command=apply_threshold(X_train))
# button = tkinter.Button(root, text="Segmentar Teste", command=apply_threshold(X_test))

#Chamada de componentes
label_1 = tkinter.Label(root, image=image)

scale_1.pack()
scale_2.pack()
scale_3.pack()
# button.pack()

label_1.pack()

# Criacao de variaveis
X_train,X_test,y_train,y_test = read_data()

#Invertendo imagens de treino
print(X_train[2])
X_train_invert = cv2.flip(X_train[2], flipCode=1)
print(X_train_invert)

#Equalizando histogramas
print(X_train[2])
X_train_equalize = cv2.equalizeHist(X_train[2])
print(X_train_equalize)

#Aplicando segmentacao
X_train = apply_threshold(X_train)
X_test = apply_threshold(X_test)

root.mainloop()
