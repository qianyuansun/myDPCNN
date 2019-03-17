from torch.utils import data
import pandas as pd

class TextDataset(data.Dataset):

    def __init__(self, path, vocab):

        train = pd.read_csv(path, header = None, skiprows=1)
        self.train_labels = train[1]
        self.train_set = train[2]

        self.train_set = self.train_set.apply(lambda row: row.strip("['").strip("']"))
        self.train_set = self.train_set.apply(lambda row: row.replace("', '", ","))

        data_array = list()

        for i in self.train_set:
            row_array = list()
            for word in i.split(","):
                if word in vocab.keys():
                    row_array.append(float(vocab[word]))
                else:
                    row_array.append(float(0))
            data_array.append(row_array)

        self.train_set = data_array
        self.word_num = len(vocab) + 1

    def __getitem__(self, index):
        return self.train_set[index], self.train_labels[index]

    def __len__(self):
        return len(self.train_set)


