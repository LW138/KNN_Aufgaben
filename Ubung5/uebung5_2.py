import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader, random_split

from Ubung5.uebung5_1 import encode_data, fill_missing_values, normalisieren, get_prepared_features


# a dataset class must implement the __init__, __len__, and __getitem__ methods
class MyDataset(Dataset):
    def __init__(self, data_file_path, transform=None, target_transform=None):
        # read data from csv file to pands dataframe
        self.data = pd.read_csv(data_file_path, sep=",", header=None)

        # set the column names to string
        self.data.columns = self.data.columns.astype(str)
        #  get all columns except the last one as features and the last column as labels
        self.features = self.data.iloc[:, :-1]
        self.labels = self.data.iloc[:, -1]

        # prepare the features
        self.features = get_prepared_features(self.features)

        # if the target is also categorical, encode it
        if self.labels.dtype == 'object':
            self.labels = LabelEncoder().fit_transform(self.labels)

        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get the data and label at the given index
        item_features = self.features.iloc[idx].values.astype('float32')
        item_label = self.labels[idx]

        # if a transform function is defined, use it to transform the feature variables
        if self.transform:
            item_features = self.transform(item_features)

        # if a target_transform function is defined, use it to transform the target variable
        if self.target_transform:
            item_label = self.target_transform(item_label)

        return item_features, item_label


if __name__ == "__main__":
    my_dataset = MyDataset("adult/adult.data", transform=None, target_transform=None)

    # split the dataset into training, validation and test sets
    train_len = int(0.8 * len(my_dataset))
    valid_len = int(0.1 * len(my_dataset))
    test_len = len(my_dataset) - train_len - valid_len

    # random_split is used to split the dataset into the three sets
    train_dataset, valid_dataset, test_dataset = random_split(my_dataset, [train_len, valid_len, test_len])

    # DataLoader is used to create a batched dataset
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    train_features, train_labels = next(iter(train_loader))
    print("Train Features: ", train_features.size())
    print("Label: ", train_labels.size())
    print("Train Features: ", train_features)
    print("Label: ", train_labels)