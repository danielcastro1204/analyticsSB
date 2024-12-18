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
    fs = 20  
    accSignalX = [reading.y for reading in signal.accelerometer]
    accSignalX = np.array(accSignalX)-np.mean(accSignalX)
    spectrum = fft(accSignalX)
    freqs = fftfreq(len(accSignalX), 1/fs)
    half_spectrum = np.abs(spectrum[:len(spectrum) // 2])
    half_freqs = freqs[:len(freqs) // 2]

   
    dominant_idx = np.argmax(half_spectrum)  # Índice del pico máximo
    dominant_frequency = half_freqs[dominant_idx]
    dominant_amplitude = half_spectrum[dominant_idx]

    min_frequency_threshold = 1  # Hz
    filtered_spectrum = half_spectrum[half_freqs > min_frequency_threshold]
    filtered_freqs = half_freqs[half_freqs > min_frequency_threshold]
    filtered_dominant_idx = np.argmax(filtered_spectrum)
    filtered_dominant_frequency = filtered_freqs[filtered_dominant_idx]

    spectral_entropy = compute_spectral_entropy(half_spectrum)

    return {"frequency":freqs.tolist(), "spectrum":half_spectrum.tolist(), "domain":dominant_frequency, 
            "entropy": spectral_entropy, "filtered": filtered_dominant_frequency}



def compute_spectral_entropy(spectrum):
  
    power_spectrum = spectrum ** 2
    
 
    normalized_spectrum = power_spectrum / np.sum(power_spectrum)
    
   
    entropy = -np.sum(normalized_spectrum * np.log2(normalized_spectrum + 1e-12))  # Evitar log(0)
    return entropy
