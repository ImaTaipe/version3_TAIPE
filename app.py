import streamlit as st
import numpy as np
import requests
from io import BytesIO

from PIL import Image

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "ImaTaipe/dataset-gatos-perros/main"
)

@st.cache_resource
def cargar_dataset():

    X = []
    y = []

    # GATOS

    for i in range(1, 14):

        url = f"{BASE_URL}/gatos/gato{i}.jpg"

        try:

            respuesta = requests.get(
                url,
                timeout=10
            )

            imagen = Image.open(
                BytesIO(respuesta.content)
            ).convert("RGB")

            imagen = imagen.resize(
                (64, 64)
            )

            X.append(
                np.array(imagen)
            )

            y.append(0)

        except:

            pass

    # PERROS

    for i in range(1, 14):

        url = f"{BASE_URL}/perros/perro{i}.jpg"

        try:

            respuesta = requests.get(
                url,
                timeout=10
            )

            imagen = Image.open(
                BytesIO(respuesta.content)
            ).convert("RGB")

            imagen = imagen.resize(
                (64, 64)
            )

            X.append(
                np.array(imagen)
            )

            y.append(1)

        except:

            pass

    X = np.array(X) / 255.0
    y = np.array(y)

    return X, y


@st.cache_resource
def entrenar_modelo():

    X, y = cargar_dataset()

    st.write(
        f"Dataset cargado: {len(X)} imágenes"
    )

    modelo = Sequential([

        Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(64, 64, 3)
        ),

        MaxPooling2D((2, 2)),

        Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),

        MaxPooling2D((2, 2)),

        Flatten(),

        Dense(
            128,
            activation="relu"
        ),

        Dense(
            2,
            activation="softmax"
        )
    ])

    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    modelo.fit(
        X,
        y,
        epochs=20,
        verbose=0
    )

    return modelo

st.title(
    "Clasificador de Gatos y Perros"
)
st.subheader(
    "Versión 3 - Dataset en la nube (GitHub)"
)
st.write(
    """
    El dataset se descarga automáticamente
    desde GitHub cada vez que se ejecuta
    la aplicación.
    """
)
modelo = entrenar_modelo()
archivo = st.file_uploader(
    "Seleccione una imagen",
    type=["jpg", "jpeg", "png"]
)
if archivo is not None:
    imagen = Image.open(
        archivo
    ).convert("RGB")
    imagen = imagen.resize(
        (64, 64)
    )
    st.image(
        imagen,
        caption="Imagen cargada",
        width=300
    )
    arr = np.array(
        imagen
    ) / 255.0
    arr = arr.reshape(
        1,
        64,
        64,
        3
    )
    pred = modelo.predict(
        arr,
        verbose=0
    )
    clase = np.argmax(pred)
    confianza = np.max(pred) * 100
    if clase == 0:
        st.success(
            f"Resultado: GATO ({confianza:.2f}%)"
        )
    else:
        st.success(
            f"Resultado: PERRO ({confianza:.2f}%)"
        )