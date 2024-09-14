import numpy as np
import matplotlib.pyplot as plt

# Ścieżka zapisu spektrogramów
save_path = 'G:/PracaMagisterska/Praca/img/example_of_spectograms/'

# Lista plików do wczytania
files = [
    ('Mono_Continous_Audio_BMF',
     'G:/PracaMagisterska/Dane/Mono_Continous_Audio/BMF/mel_22050_1024_512/G61-45105-BMF-S7.8T8.4-20598.npy'),
    ('Mono_Discret_Audio_OD1',
     'G:/PracaMagisterska/Dane/Mono_Discret_Audio/OD1/mel_22050_1024_512/G61-56211-OD1-D8-20617.npy'),
    ('Poly_Continuous_Audio_TS9',
     'G:/PracaMagisterska/Dane/Poly_Continuous_Audio/TS9/mel_22050_1024_512/P64-43110-TS9-D2.9T2.1-41185.npy'),
    ('Poly_Discret_Audio_FFC',
     'G:/PracaMagisterska/Dane/Poly_Discret_Audio/FFC/mel_22050_1024_512/P64-44140-FFC-F10-41231.npy')
]

# Iterowanie przez listę plików
for name, file_path in files:
    # Wczytaj mel-spektrogram
    mel_spectrogram = np.load(file_path)

    # Generowanie wykresu mel-spektrogramu
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.title(f'Mel-Spectrogram: {name}')
    plt.xlabel('Czas')
    plt.ylabel('Częstotliwość Mel')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

    # Zapisanie wykresu jako plik PNG
    plt.savefig(f'{save_path}{name}.png', format='png')
    plt.close()

print("Spektrogramy zostały zapisane.")
