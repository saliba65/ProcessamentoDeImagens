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
import time
import tqdm
import numpy as np 
import tensorflow as tf

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

def start_program() -> None:
    # Ler diretorio e atribir variaveis 
    X_train, X_test, y_train, y_test = read_data()

    # Chamada funcao de inversao 
    X_train, y_train = flip_data(X_train, y_train)

    # Chamada funcao de equalizar
    X_train, y_train = equalize_data(X_train, y_train)

    # Aplicando segmentacao
    X_train = apply_threshold(X_train)
    X_test = apply_threshold(X_test)

    # Conversao para classificacao binária
    y_train_binary = convert_binary(y_train)
    y_test_binary = convert_binary(y_test)

    '''
    # Transformacao para numpy array 
    X_train = np.array(X_train, dtype="float32")
    X_train = np.expand_dims(X_train, -1)
    X_test = np.array(X_test, dtype="float32")
    X_test = np.expand_dims(X_test, -1)
    y_train = np.array(y_train)
    y_train = np.expand_dims(y_train, -1)
    y_train = np.expand_dims(y_train, -1)
    y_train = np.expand_dims(y_train, -1)
    #y_train = tf.keras.utils.to_categorical(y_train, 4)
    y_test = np.array(y_test)
    #y_test = tf.keras.utils.to_categorical(y_test, 4)
    y_train_binary = np.array(y_train_binary, dtype="float32")
    y_test_binary = np.array(y_test_binary, dtype="float32")
    '''

    # Rede Neural
    # conv_next(X_train, X_test, y_train, y_test)

    conv(X_train, X_test, y_train_binary, y_test_binary, 2)
    conv(X_train, X_test, y_train, y_test, 4)

# Verificar se a imagem pertence ao conjunto de testes
def is_test(file: str) -> bool:
    id = int(file.split('(')[1].strip(').png'))

    answer = False
    
    if id % 4 == 0:
        answer = True

    return answer

# Ler diretorios de mamografias separando treino e teste
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
                for file in os.listdir(f'./mamografias/{folder}')[:10]:
                    if file.endswith('.png'):
                        image = cv2.imread(f'./mamografias/{folder}/{file}', cv2.IMREAD_GRAYSCALE)
                        # Resize imagens para rede neural
                        image = cv2.resize(image, (224, 224))

                        #Atribuir a conjunto de testes
                        if is_test(file):
                            X_test.append(image)
                            y_test.append(y_dict[folder[0]])
                        #Atribuir a conjunto de treino
                        else:
                            X_train.append(image)
                            y_train.append(y_dict[folder[0]])

    return (X_train, X_test, y_train, y_test)

# Invertendo imagens de treino
def flip_data(X_train: list, y_train: list) -> tuple:
    X_transformed = X_train.copy()
    y_transformed = y_train.copy()
    
    for X, y in zip(X_train, y_train):
        X_transformed.append(cv2.flip(X, flipCode=1))
        y_transformed.append(y)

    return (X_transformed, y_transformed)

# Equalizando histogramas
def equalize_data(X_train: list, y_train: list) -> tuple:
    X_transformed = X_train.copy()
    y_transformed = y_train.copy()
    
    for X, y in zip(X_train, y_train):
        X_transformed.append(cv2.equalizeHist(X))
        y_transformed.append(y)

    return (X_transformed, y_transformed)
    
# Segmentacao automatica de imagens do diretorio
def apply_threshold(X_array: list) -> list:
    X_transformed = []
    for X in X_array:
        # Aplicacao de segmentacao binaria em cada item recebido
        X_transformed.append(cv2.threshold(X, 7, 255, cv2.THRESH_BINARY)[1])

    return X_transformed

# Conversao para classificacao binária
def convert_binary(y_array: list) -> list:
    y_transformed = []
    
    y_dict = {
        0: 0,
        1: 0,
        2: 1,
        3: 1
    }

    for y in y_array:
        y_transformed.append(y_dict[y])

    return y_transformed

'''
# Aplicacao da rede neural ConvNext 
def conv_next(X_train: np.array, X_test: np.array, y_train: np.array, y_test: np.array) -> None:
    model = tf.keras.applications.ConvNeXtBase(
        model_name="convnext_base",
        include_top=False,
        include_preprocessing=False,
        weights=None,
        input_tensor=tf.keras.layers.Input(shape=(224, 224, 1)),
        input_shape=None,
        pooling=None,
        classes=4,
        classifier_activation="softmax"
    )

    print('convnext')

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    print('compile')

    model.fit(X_train, y_train, verbose=1)

    print('fit')
'''

# Rede neural convolucional 
def conv(X_train: list, X_test: list, y_train: list, y_test: list, classes: int) -> None:
    start = time.time()
    
    # Conversao para numpy array
    X_train = np.array(X_train, dtype=object)
    X_test = np.array(X_test, dtype=object)
    y_train = np.array(y_train, dtype=object)
    y_test = np.array(y_test, dtype=object)

    # Normalizacao dos dados ( entre 0 e 1 )
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # Adicionar uma dimensao aos np arrays para atender especificidades da rede
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)

    # Criacao de classes para o keras
    y_train = tf.keras.utils.to_categorical(y_train, classes)
    y_test = tf.keras.utils.to_categorical(y_test, classes)
    
    # Modelo de rede neural convolucional 
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(224, 224, 1)),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(classes, activation="softmax")
        ]
    )

    # Compilar o modelo 
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=[
        "accuracy",
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.SpecificityAtSensitivity(0.5)
    ])

    # Realizar ajuste dos dados
    model.fit(X_train, y_train, batch_size=16, epochs=16, validation_split=0.1)

    # Captacao de informacoes para exibir na tela
    score = model.evaluate(X_test, y_test, verbose=0)

    f1_score = 2 * (score[2] * score[3]) / (score[2] + score[3])

    # Modelo Binario
    if classes == 2:
        text = f'''
            Binário (2 Classes):\n
            Loss: {score[0]:.4f}
             Acurácia: {score[1]:.4f}
             F1 Score: {f1_score:.4f}
             Precisao: {score[2]:.4f}
             Sensibilidade: {score[3]:.4f}
             Especificidade: {score[4]:.4f}
             Tempo: {time.time() - start:.2f}s
        '''
        
        label_2.configure(text=text)

        label_2.text = text
        
    # Modelo 4 classes
    else:
        text = f'''
            4 Classes:\n
            Loss: {score[0]:.4f}
             Acurácia: {score[1]:.4f}
             F1 Score: {f1_score:.4f}
             Precisao: {score[2]:.4f}
             Sensibilidade: {score[3]:.4f}
             Especificidade: {score[4]:.4f}
             Tempo: {time.time() - start:.2f}s
        '''
        
        label_3.configure(text=text)

        label_3.text = text


# Usando tkinter para leitura de imagens
# Permitindo buscar arquivos png e tiff na biblioteca 
root = tkinter.Tk()
root.geometry('1024x512')
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
button = tkinter.Button(root, text="Treinar Modelo", command=start_program)

#Chamada de componentes
label_1 = tkinter.Label(root, image=image)
label_2 = tkinter.Label(root, text='')
label_3 = tkinter.Label(root, text='')

scale_1.pack()
scale_2.pack()
scale_3.pack()
button.pack()


label_2.pack()
label_3.pack()
label_1.pack()

root.mainloop()
