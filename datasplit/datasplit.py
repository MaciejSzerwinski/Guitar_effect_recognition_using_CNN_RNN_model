import numpy as np
import torch
import logging
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# from https://palikar.github.io/posts/pytorch_datasplit/
class DataSplit:

    def __init__(self, dataset, test_train_split=0.8, val_train_split=0.1, shuffle=False):
        self.dataset = dataset

        dataset_size = len(dataset)     # Całkowita liczba próbek w zbiorze danych
        self.indices = list(range(dataset_size))
        test_split = int(np.floor(test_train_split * dataset_size))     # to liczba próbek, które trafią do zbioru treningowego

        if shuffle:
            np.random.shuffle(self.indices)

        train_indices, self.test_indices = self.indices[:test_split], self.indices[test_split:]     # podział próbek na testowe i trenowane
        train_size = len(train_indices)
        validation_split = int(np.floor((1 - val_train_split) * train_size))

        self.train_indices, self.val_indices = train_indices[:validation_split], train_indices[validation_split:]   # podział próbek na trenowane i walidacyjne

        self.train_sampler = SubsetRandomSampler(self.train_indices)       # Samplery do losowego wybierania próbek z odpowiednich zestawów.
        self.val_sampler = SubsetRandomSampler(self.val_indices)
        self.test_sampler = SubsetRandomSampler(self.test_indices)

    def get_train_split_point(self):        # Zwraca sumę próbek części treningowej i walidlacyjne
        return len(self.train_sampler) + len(self.val_indices)

    def get_validation_split_point(self):       # Zwraca ilość próbek treningowych
        return len(self.train_sampler)

    def get_split(self, batch_size=50, num_workers=0):  # batch_size: Liczba próbek w jednej partii, num_workers: Liczba wątków do ładowania danych
        logging.debug('Initializing train-validation-test dataloaders')
        self.train_loader = self.get_train_loader(
            batch_size=batch_size, num_workers=num_workers)
        self.val_loader = self.get_validation_loader(
            batch_size=batch_size, num_workers=num_workers)
        self.test_loader = self.get_test_loader(
            batch_size=batch_size, num_workers=num_workers)
        return self.train_loader, self.val_loader, self.test_loader

    def get_train_loader(self, batch_size=50, num_workers=0):
        logging.debug('Initializing train dataloader')
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, sampler=self.train_sampler, shuffle=False, num_workers=num_workers)
        return self.train_loader

    def get_validation_loader(self, batch_size=50, num_workers=0):
        logging.debug('Initializing validation dataloader')
        self.val_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, sampler=self.val_sampler, shuffle=False, num_workers=num_workers)
        return self.val_loader

    def get_test_loader(self, batch_size=50, num_workers=0):
        logging.debug('Initializing test dataloader')
        self.test_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, sampler=self.test_sampler, shuffle=False, num_workers=num_workers)
        return self.test_loader