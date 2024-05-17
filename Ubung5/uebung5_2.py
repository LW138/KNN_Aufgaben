import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch import tensor
from torch.utils.data import Dataset, DataLoader, random_split

from Ubung5.uebung5_1 import encode_data, fill_missing_values, normalisieren


# a dataset class must implement the __init__, __len__, and __getitem__ methods
class MyDataset(Dataset):
    def __init__(self, data_file_path, transform=None, target_transform=None):
        self.data = pd.read_csv(data_file_path, sep=",", header=None)

        self.features = self.data.iloc[:, :-1] # all rows, all columns except the last one
        self.labels = self.data.iloc[:, -1] # all rows, only the last column

        # use encode_data function from exercise 5.1
        self.features = encode_data(self.features)

        # ff the target is also categorical, encode it
        if self.labels.dtype == 'object':
            self.labels = LabelEncoder().fit_transform(self.labels)

        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item_data = self.features.iloc[idx].values.astype('float32')
        item_label = self.labels[idx]
        if self.transform:
            item_data = self.transform(item_data)
        if self.target_transform:
            item_label = self.target_transform(item_label)

        return item_data, item_label



my_dataset = MyDataset("adult/adult.data", transform=None, target_transform=None)

train_len = int(0.8 * len(my_dataset))
valid_len = int(0.1 * len(my_dataset))
test_len = len(my_dataset) - train_len - valid_len

train_dataset, valid_dataset, test_dataset = random_split(my_dataset, [train_len, valid_len, test_len])

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

train_features, train_labels = next(iter(train_loader))
print("Train Features: ", train_features.size())
print("Label: ", train_labels.size())
label = train_labels[0]
print("Label: ", label)