from fastapi import FastAPI
import numpy as np
from models import Item, SensorInput
from scipy.fft import fft, fftfreq

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hola Mundo"}

@app.post("/items/")
def create_item(item: Item):
    return {"name": item.name, "description": item.description}


@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}


@app.post("/fft")
def fftAsAService(signal:SensorInput):
    fs = 44  # frecuencia de muestreo
    accSignalX = [reading.y for reading in signal.accelerometer]
    accSignalX = np.array(accSignalX)-np.mean(accSignalX)
    spectrum = fft(accSignalX)
    freqs = fftfreq(len(accSignalX), 1/fs)
    half_spectrum = np.abs(spectrum[:len(spectrum) // 2])
    half_freqs = freqs[:len(freqs) // 2]

    ##Frecuencia dominante
    dominant_index = np.argmax(half_spectrum)
    dominant_frequency = half_freqs[dominant_index]
    spectral_entropy = compute_spectral_entropy(half_spectrum)

    return {"frequency":freqs.tolist(), "spectrum":half_spectrum.tolist(), "domain":dominant_frequency, "entropy": spectral_entropy}

#Sacar fft y sacar las freqs

def compute_spectral_entropy(spectrum):
    # Espectro de potencia
    power_spectrum = spectrum ** 2
    
    # Normalizar el espectro de potencias
    normalized_spectrum = power_spectrum / np.sum(power_spectrum)
    
    # Calcular la entropía
    entropy = -np.sum(normalized_spectrum * np.log2(normalized_spectrum + 1e-12))  # Evitar log(0)
    return entropy

