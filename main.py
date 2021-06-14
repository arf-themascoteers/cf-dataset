from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

class Censius2FahrenheitDataset(Dataset):
    def __init__(self, is_train):
        self.is_train = is_train
        self.celsius_values = np.linspace(-50, 50, 1000)
        self.fahrenheit_values = self.celsius_to_fahrenheit(self.celsius_values)
        self.data_indices = None
        if is_train:
            self.data_indices = [i for i in range(len(self.celsius_values)) if i%10 != 0]
        else:
            self.data_indices = [i for i in range(len(self.celsius_values)) if i%10 == 0]

    def celsius_to_fahrenheit(self, celsius):
        return (celsius * 9 / 5) + 32

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        index_in_full_set = self.data_indices[idx]
        return self.celsius_values[index_in_full_set], self.fahrenheit_values[index_in_full_set]

if __name__ == "__main__":
    dataset = Censius2FahrenheitDataset(is_train=True)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False)

    for celsius, fahrenheit in dataloader:
        print(celsius, fahrenheit)
        break

    dataset = Censius2FahrenheitDataset(is_train=False)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

    for celsius, fahrenheit in dataloader:
        print(celsius, fahrenheit)
        break