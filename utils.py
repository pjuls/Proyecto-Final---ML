import opensmile
import numpy as np
import os
import numpy as np
import pandas as pd
import librosa
import torch

from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2Model

#1
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)

#2
processor = None
model = None
def load_wav2vec():
    global processor, model

    if processor is None:
        processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base"
        )

    if model is None:
        model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )
        model.eval()



def extract_egemaps(audio_path):
    """
    Recibe la ruta de un archivo .wav y devuelve
    un vector de aproximadamente 88 features eGeMAPS.
    """
    features = smile.process_file(audio_path)
    #no devuelve un vector, devuelve un DataFrame de pandas.

    # features es un DataFrame de una sola fila
    return features.values[0]



def extract_wav2vec_embedding(audio_path):

    load_wav2vec()
    # Cargar audio
    waveform, sr = librosa.load(audio_path, sr=16000)

    # Preparar entrada
    inputs = processor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt"
    )

    # Forward (sin gradientes)
    with torch.no_grad():

        outputs = model(**inputs)

    # Salida del último transformer
    hidden_states = outputs.last_hidden_state

    # Promedio temporal
    embedding = hidden_states.mean(dim=1)

    return embedding.squeeze().numpy() # devuelve vector de (768,)


def build_feature_dataframe(df, n):
    """
    Convierte todos los audios de un DataFrame
    en un DataFrame de features.
    """

    feature_vectors = []

    if n==1:
        for audio in df["file"]:
            feature_vectors.append(extract_egemaps(audio))
    else:
        for audio in df["file"]:
            feature_vectors.append(
                extract_wav2vec_embedding(audio)
            )        

    feature_vectors = np.array(feature_vectors)


    if n==1:
        # Obtenemos los nombres reales de las features
        feature_names = smile.process_file(df.iloc[0]["file"]).columns
    else:
        feature_names = [
            f"wav2vec_{i}"
            for i in range(feature_vectors.shape[1])
        ]         


    feature_df = pd.DataFrame(
        feature_vectors,
        columns=feature_names
    )

    # Agregamos información útil al csv (desp cuando saquemos el x no lo leemos .. )
    feature_df["emotion"] = df["emotion"].values
    feature_df["actor"] = df["actor"].values
    feature_df["gender"] = df["gender"].values

    return feature_df

