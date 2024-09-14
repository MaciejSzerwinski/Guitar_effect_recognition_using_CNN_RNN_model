import torch
import numpy as np
import librosa
import os
import csv
from torch.utils.data import Dataset

class FxDataset(Dataset): # Redone by me
    def __init__(self, 
                    root, 
                    excl_folders=None, 
                    spectra_folder=None,
                    processed_settings_csv='proc_settings.csv',
                    max_num_settings=6,
                    transform=None):

        #Opis ścieżek oraz do konkretnych danych#
        # paths and co
        self.root = root                        # root folder
        self.excl_folders = excl_folders        # array of folders excluded from the data
        self.spectra_folder = spectra_folder    # folder containing spectral features
        self.processed_settings_csv = processed_settings_csv
        
        # processing
        self.max_num_settings = max_num_settings
        self.transform = transform
        
        # fx related data structures
        self.fx_to_label = {}                   # dict {fx_name(str):fx_label(int)}
        self.fx_to_label_oh = {}                # dict {fx_name(str):fx_label_onehot(bool)} PEWNIE BŁĄD dict {fx_name(str):fx_label_onehot(list)} Ponieważ to będzie lista słów bitowych w poniższych przykładach to samo
        self.label_to_fx = {}                   # dict {fx_label(int):fx_name(str)}
        self.label_oh_to_fx = {}                # dict {fx_label_onehot(list):fx_name(str)}
        
        self.audiosample_to_label = {}          # dict {filename(str):fx_label(int)}
        self.audiosample_to_label_oh = {}       # dict {filename(str):fx_label_onehot(bool)} BŁĄD dict {filename(str):fx_label_onehot(list)}

        self.audiosamples_labels = []           # array [(filename(str), fx_label(int))]
        self.audiosamples_labels_oh = []         # array [(filename(str), fx_label_onehot(bool))] BŁĄD array [(filename(str), fx_label_onehot(list))]
        
        # settings related data structures
        self.audiosample_to_settings = {}       # dict {filename(str):settings(float)}
        self.audiosamples_settings = []         # array [(filename, settings)]
        
        # for conditioning
        self.mel_shape = ()
        self.num_fx = 0
    

    def __len__(self):  # Zwraca liczbę próbek audio w zbiorze danych. # Redone by me
        return len(self.audiosamples_labels)
    

    def __getitem__(self, index): # Ta metoda zwraca próbkę danych na podstawie indeksu.
        filename, label = self.audiosamples_labels[index]
        _, settings = self.audiosamples_settings[index]
        folder = self.label_to_fx[label]
        try:
            X = np.load("%s/%s/%s/%s.npy" % (self.root, folder, self.spectra_folder, filename))
        except:
            print("error __getitem__:", filename)
            return None
        
        if self.transform:
            X = self.transform(X)
        
        label = torch.tensor(label) # Konwertowanie zmiennych na tensory PyTorch. Przygotowanie danych do modelowania i uczenia maszynowego
        settings = torch.tensor(settings)
        
        return X, label, settings, filename, index
    

    def init_dataset(self):
        self.init_fx_labels_dict() # Zawiera nazwy foldedrów i przypisane do niego etykiety
        self.init_fx_labels_oh_dict() # Stwożenie słownika zawierającego nazwy folderów oraz odpowiadającej do nich etkiet w formie binarnej metodą onhot
        self.init_audiosamples_labels() # Stworzenie tablicy w której zawartę będą nazwy próbek oraz ich etykiety
        self.init_audiosamples_labels_oh()  # Stworzenie tablicy z zawartością nazw próbek oraz ich etykiet w formie binanrnej
        self.init_audiosample_to_label_dict()   # Stworzenie słownika z nazwami oraz etykietami danych próbek dźwiękowych
        self.init_audiosample_to_label_oh_dict()    # Stworzenie słownika z nazwami oraz etykietami w reprezentacji binarnej
        self.init_audiosample_to_settings_dict()    # Stworzenie słownika z nazwą próbki oraz jej ustawieniami
        self.init_audiosamples_settings()   # Stworzenie tablicy z nazwą próbki oraz jej ustawieniami
        
        X = self.__getitem__(0)
        if X:
            self.mel_shape = X[0].shape # Sprawdzamy rozmiar próbki
            self.num_fx = len(self.fx_to_label)
        
        
    # generates mel spectrograms and saves them in the dataset folders
    def generate_mel(self, sr=22050, n_fft=1024, hop_length=512):
        print('Generate MEL spectrograms')
        for folder in sorted(os.listdir(self.root)):
            if os.path.isdir(os.path.join(self.root, folder)):
                print('Working on folder:', folder)

                if self.spectra_folder:
                    out_path = '%s/%s/%s' % (self.root, folder, self.spectra_folder)    # root+folder= G:\PracaMagisterska\Dane\Mono_Continous_Audio
                else:
                    self.spectra_folder = '%s_%s_%s_%s' % ("mel", sr, n_fft, hop_length)
                    out_path = '%s/%s/%s' % (self.root, folder, self.spectra_folder)

                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                    print('out_path: ', out_path)    
                    for file in os.listdir('%s/%s' % (self.root, folder)):
                        if not(file.startswith("._")) and file.endswith(".wav"):
                            filename = file[:-4]
                            try:
                                audio_file, sr = librosa.load("%s/%s/%s.wav" % (self.root, folder, filename), sr=sr)
                                mel_file = librosa.feature.melspectrogram(audio_file, sr=sr, n_fft=n_fft, hop_length=hop_length)
                                np.save(file=('%s/%s' % (out_path, filename)), arr=mel_file)
                            except:
                                print('error generate_mel:', file)
                else:
                    print('out_path: ', out_path, ' already exists!')

        print('Generate MEL spectrogram complete')
        print('folder: %s' % self.spectra_folder)
    
    # initialise fx_to_labels and labels_to_fx dictionaries
    def init_fx_labels_dict(self): # Redone by me
        i = 0
        for folder in sorted(os.listdir(self.root)):
            if os.path.isdir(os.path.join(self.root, folder)):
                if not folder in self.excl_folders:
                    self.fx_to_label[folder] = i
                    self.label_to_fx[i] = folder
                    i += 1
                
    # intialise fx_to_label_oh and label_oh_to_fx dictionaries
    def init_fx_labels_oh_dict(self):
        self.excl_folders = self.excl_folders if self.excl_folders is not None else []
        valid_folders = [folder for folder in sorted(os.listdir(self.root)) if
                         os.path.isdir(os.path.join(self.root, folder)) and folder not in self.excl_folders]
        bits = len(valid_folders)

        if bits <= 0:
            raise ValueError("The calculated number of bits is non-positive. Check your dataset structure.")

        for i, folder in enumerate(valid_folders):
            oh_array = [0] * bits
            oh_array[i] = 1

            self.fx_to_label_oh[folder] = oh_array
            self.label_oh_to_fx[''.join(str(e) for e in oh_array)] = folder
                
    # initialise audiosamples_labels array
    def init_audiosamples_labels(self):     # Redone by me
        if not(self.fx_to_label or self.label_to_fx): # Warunek jeśli nie istnieje słownik foldlerów i ich etykiety lub odwrotnie dla danego obiektu te klasy to wywoływana jest funkcja init
            self.init_fx_labels_dict()
        for folder in sorted(os.listdir(self.root)):
            if os.path.isdir(os.path.join(self.root, folder)):
                if not folder in self.excl_folders:
                    for file in os.listdir('%s/%s/%s' % (self.root, folder, self.spectra_folder)):
                        if not(file.startswith("._")) and file.endswith(".npy"):
                            filename = file[:-4] # Usuwanie rozszerzenia .npy
                            label = self.fx_to_label[folder]
                            self.audiosamples_labels.append((filename,label))

    # initialise audiosamples_labels_oh array    
    def init_audiosamples_labels_oh(self):      # Redone by me
        if not(self.fx_to_label_oh or self.label_oh_to_fx):
            self.init_fx_labels_oh_dict()
        for folder in sorted(os.listdir(self.root)):
            if os.path.isdir(os.path.join(self.root, folder)):                
                if not folder in self.excl_folders:
                    for file in os.listdir('%s/%s/%s' % (self.root, folder, self.spectra_folder)):
                        if not(file.startswith("._")) and file.endswith(".npy"):
                            filename = file[:-4]
                            label = self.fx_to_label_oh[folder]
                            self.audiosamples_labels_oh.append((filename,label))

    # initialise audiosample_to_label dictionary
    def init_audiosample_to_label_dict(self):   # Redone by me
        if not(self.audiosamples_labels):       #Sprawdzenie czy audiosamples_labels istnieje jeżeli nie to trzeba zainicjować
            self.init_audiosamples_labels()
        for sample in self.audiosamples_labels:
            samplename, samplelabel = sample
            self.audiosample_to_label[samplename] = samplelabel     # W tym miejscu tworzony jest słownik z powstałej wcześniej listy próbek i i ch etykiet
    
    # initialise audiosample_to_label_oh dictionary
    def init_audiosample_to_label_oh_dict(self):        # Redone by me
        if not(self.audiosamples_labels_oh):
            self.init_audiosamples_labels_oh()
        for sample in self.audiosamples_labels_oh:
            samplename, samplelabel = sample
            self.audiosample_to_label_oh[samplename] = samplelabel
    
    # initialise audiosample_to_settings dictionary
    def init_audiosample_to_settings_dict(self):        # Redone by me
        for folder in sorted(os.listdir(self.root)):
            if os.path.isdir(os.path.join(self.root, folder)):
                if not folder in self.excl_folders:
                    with open("%s/%s/%s" % (self.root, folder, self.processed_settings_csv), mode='r') as file:
                        rdr = csv.reader(file)
                        next(rdr)
                        for row in rdr:
                            filename, fx, *settings = row
                            filename = filename[:-4]
                            blanks = [-1.0] * (self.max_num_settings-len(settings))
                            settings = settings + blanks
                            settings = [float(i) for i in settings]     # Konwersja danych ponieważ wczytywane dane z pliku CSV traktowane są jako zmienna str dlatego trzeba to przekonwertować na float
                            self.audiosample_to_settings[filename] = settings
    

    # initialise audiosample_settings array
    def init_audiosamples_settings(self):
        if not(self.audiosample_to_settings):       # Redone by me
            self.init_audiosample_to_settings_dict()
        for folder in sorted(os.listdir(self.root)):
            if os.path.isdir(os.path.join(self.root, folder)):
                if not folder in self.excl_folders:
                    for file in os.listdir('%s/%s/%s' % (self.root, folder, self.spectra_folder)):
                        if not(file.startswith("._")) and file.endswith(".npy"):
                            filename = file[:-4]
                            settings = self.audiosample_to_settings[filename]
                            self.audiosamples_settings.append((filename,settings))

                            
            
    # def init_audiosamples_fx_settings_oh(self):
    #     for folder in os.listdir(self.root):
    #         if os.path.isdir(os.path.join(self.root, folder)):
    #             if folder != 'NoFX':
    #                 with open("%s/%s/%s" % (self.root, folder, "proc_settings.csv"), mode='r') as file:
    #                     rdr = csv.reader(file)
    #                     next(rdr)
    #                     for row in rdr:
    #                         filename, fx, level, tone, gain = row
    #                         l=self.level_to_label_oh[float(level)]
    #                         t=self.tone_to_label_oh[float(tone)]
    #                         g=self.gain_to_label_oh[float(gain)]
    #                         self.audiosamples_fx_settings_oh.append((filename,l+t+g))
                        
    
    # def settings_binary_to_string(self, bin_str):
    #     if len(bin_str) != 9:
    #         print('error settings_binary_to_string')
    #         return
    #     l = self.label_oh_to_level[bin_str[0:3]]
    #     t = self.label_oh_to_tone[bin_str[3:6]]
    #     g = self.label_oh_to_gain[bin_str[6:9]]
    #     return l+t+g