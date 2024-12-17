import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utilities.constants import dataset_prompt


class CustomerDataset(Dataset):
    def __init__(self, data_path) -> None:
        super(CustomerDataset, self).__init__()

        df = pd.read_csv(data_path)
        query_1 = df['query1'].to_list()
        query_2 = df['query2'].to_list()
        labels = df['label'].to_list()

        self.final_queries = [dataset_prompt.format(q_1=q_1, q_2=q_2) for q_1, q_2 in zip(query_1, query_2)]
        self.labels = labels

        assert len(self.final_queries) == len(self.labels)

    def __len__(self):
        return len(self.final_queries)

    def __getitem__(self, index):
        query = self.final_queries[index]
        label = self.labels[index]

        return query, label


def load_data(data_path, tokenizer, batch_size=32, mode='train'):
    def collate_fn(batch):
        queries = [item[0] for item in batch]  # the input for batch_encode_plus() must be a list
        labels = torch.tensor([item[1] for item in batch], dtype=torch.float16)

        tokenizer_batch = tokenizer.batch_encode_plus(queries,
                                                      truncation=True,
                                                      padding="longest",
                                                      max_length=256,
                                                      return_attention_mask=True,
                                                      return_tensors='pt')

        input_ids = tokenizer_batch['input_ids']
        attention_mask = tokenizer_batch['attention_mask']

        return input_ids, attention_mask, labels

    if_shuffle = True if mode == 'train' else False

    dataset = CustomerDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=if_shuffle, collate_fn=collate_fn)

    return dataloader


if __name__ == '__main__':
    demo_dataset = CustomerDataset(r"D:\CSCI544_project_code\binary_classifier\dataset\Sample_data.csv")
    print(demo_dataset[0][0], '\n')
    print('label:', demo_dataset[0][1])
    print()
