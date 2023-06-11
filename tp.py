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
# from tensorflow.keras import layers
# from tensorflow import keras
import tensorflow as tf
# import matplotlib.pyplot as plt
# import tensorflow_addons as tfa

# AUTO = tf.data.AUTOTUNE
# BATCH_SIZE=128

# class ConvNext_Block(tf.keras.Model):
    
#     """
#     Implementing the ConvNeXt block for 
    
#     Args:
#         dim: No of input channels
#         drop_path: stotchastic depth rate 
#         layer_scale_init_value=1e-6
    
#     Returns:
#         A conv block
#     """
    
#     def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6, **kwargs):
#         super(ConvNext_Block, self).__init__(**kwargs)
        
#         self.depthwise_convolution = layers.Conv2D(dim, kernel_size=7, padding="same", groups=dim )
#         self.layer_normalization = layers.LayerNormalization(epsilon=1e-6)
#         self.pointwise_convolution_1 = layers.Dense(4 * dim)
#         self.GELU = layers.Activation("gelu")
#         self.pointwise_convolution_2 = layers.Dense(dim)
#         self.gamma = tf.Variable(layer_scale_init_value * tf.ones((dim,)))
#         if drop_path>0.0:
#             self.drop_path=(tfa.layers.StochasticDepth(drop_path))
#         else:
#             self.drop_path=layers.Activation("linear")
        

#     def call(self, inputs):
#         x = inputs
#         x = self.depthwise_convolution(x)
#         x = self.layer_normalization(x)
#         x = self.pointwise_convolution_1(x)
#         x = self.GELU(x)
#         x = self.pointwise_convolution_2(x)
#         x = self.gamma * x

#         return inputs + self.drop_path(x)
    
# def patchify_stem(dims):
#     """
#     Implements the stem block of ConvNeXt
    
#     Args:
#         Dims: List of feature dimensions at each stage.
    
#     Returns:
#         feature maps after patchify operation
#     """
#     stem = keras.Sequential(
#         [layers.Conv2D(dims[0], kernel_size=4, strides=4),
#         layers.LayerNormalization(epsilon=1e-6)],
#         )
#     return stem

# def spatial_downsampling(stem,dims,kernel_size,stride):
#     """
#     Implements Spatial Downsampling of ConvNeXt
    
#     Args:
#         Dims: List of feature dimensions at each stage.
#         stem: Patchify stem output of images
#         kernel_size: Downsampling kernel_size
#         stride: Downsampling stride length
#     Returns:
#         Downsampled layers
#     """

#     ds_layers = []
#     ds_layers.append(stem)
#     for dim in dims[1:]:
#         layer = keras.Sequential(
#             [layers.LayerNormalization(epsilon=1e-6),
#             layers.Conv2D(dim, kernel_size=kernel_size, strides=stride),
#             ]
#         )
#         ds_layers.append(layer)
        
#     return ds_layers

# def ConvNext_Stages (dims,drop_path_rate,depths,layer_scale_init_value):
#     """
#     Creating stages each consiting of multiple residual blocks
    
#     Args:
#         dims: List of feature dimensions at each stage.
#         drop_path_rate: Stochastic depth rate
#         depths: Number of blocks at each stage
#         layer_scale_init_value: Init value for Layer Scale
        
#     """
#     stages = []
#     dropout_rates = [x for x in tf.linspace(0.0, drop_path_rate, sum(depths))]
#     cur = 0
#     for i in range(len(dims)):
#         stage = keras.Sequential(
#             [*[ConvNext_Block(dim=dims[i],drop_path=dropout_rates[cur + j],layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
#             ]
#         )
#         stages.append(stage)
#         cur += depths[i]
#     return stages

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

    # X_train, X_test = X_train / 255.0, X_test / 255.0 
    # train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    # #applying transformations 
    # train_data = train_data.shuffle(1024) # shuffle the images
    # train_data = train_data.map(preprocess, num_parallel_calls=AUTO)# mapping our preprocess function to train_data 
    # train_data = train_data.map(augmentation, num_parallel_calls=AUTO)# mapping our augmentation funtion to train_data
    # train_data = train_data.batch(BATCH_SIZE) #Converting train_data to batches
    # train_data = train_data.prefetch(AUTO) # using prefetch which prepares subsequent batches of data while other batches are being computed.

    # test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    # #applying transformations 
    # test_data = test_data.map(preprocess, num_parallel_calls=AUTO) # mapping our preprocess function test_data 
    # test_data = test_data.batch(BATCH_SIZE)
    # test_data = test_data.prefetch(AUTO) # using prefetch which prepares subsequent batches of data while other batches are being computed.

    # Rede Neural ConvNext
    # conv_next(X_train, X_test, y_train, y_test)

   # Rede Neural Convolucional
    # conv(X_train, X_test, y_train_binary, y_test_binary, 2)
    conv(X_train, X_test, y_train, y_test, 4)


# # convert images to float32 format 
# def preprocess (image):
#     image = tf.image.convert_image_dtype(image, tf.float32)
#     return image

# #Peform augmentations on training data
# def augmentation(image):
#     image = tf.image.resize_with_crop_or_pad(image, 40, 40) # Add 8 pixels of padding
#     image = tf.image.random_crop(image, size=[32, 32, 3]) # Random crop back to 32x32
#     image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness
#     image = tf.clip_by_value(image, 0., 1.)
#     return image

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

# Aplicacao da rede neural ConvNext 
# def conv_next(X_train: np.array, X_test: np.array, y_train: np.array, y_test: np.array) -> None:
#     model = tf.keras.applications.ConvNeXtBase(
#         model_name="convnext_base",
#         include_top=False,
#         include_preprocessing=False,
#         weights=None,
#         input_tensor=tf.keras.layers.Input(shape=(224, 224, 1)),
#         input_shape=None,
#         pooling=None,
#         classes=4,
#         classifier_activation="softmax"
#     )

#     print('convnext')

#     model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#     print('compile')

#     model.fit(X_train, y_train, verbose=1)

#     print('fit')


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
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            # tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation="relu"),
            # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
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
