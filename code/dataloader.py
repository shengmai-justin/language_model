import numpy as np
import torch


class Dataloader():
    def __init__(self, dataset, batch_size, seq_len):
        # input = whole dataset after tokenize, batch size, seq_len
        dataset_size = sum(len(tokens) for tokens in dataset)
        temp = np.array([])
        for x in dataset:
            temp = np.append(temp, x)
        temp = temp.astype(int)

        self.dataset = dataset
        self.data = torch.from_numpy(temp.reshape(batch_size, dataset_size // batch_size).transpose(1, 0))
        self.batch_size = batch_size
        self.seq_len = seq_len

    # def get_batch(self, index):
    #     unpad_batch = []
    #     pad_batch = []
    #
    #     start = index * self.batch_size
    #     end = (index + 1) * self.batch_size
    #     if end > len(self.dataset):
    #         end = len(self.dataset)
    #
    #     for i in range(start, end):
    #         unpad_batch.append(self.dataset[i])
    #
    #     for sample in unpad_batch:
    #         pad_num = self.seq_len - len(sample)
    #         if pad_num >= 0:
    #             temp = sample + [1] * pad_num
    #             pad_batch.append(temp)
    #         elif pad_num < 0:
    #             temp = sample[:pad_num]
    #             pad_batch.append(temp)
    #     x = torch.tensor(pad_batch)
    #     tgt = x.clone().detach()
    #     mask = np.triu(np.ones(self.seq_len), k=1).astype(int)
    #     mask = torch.tensor(mask)
    #     return x, tgt, mask
    def get_batch(self, index):
        batch = []
        i = 0
        start = index * self.seq_len
        end = (index + 1) * self.seq_len
        if end > len(self.dataset) - 1:
            end = len(self.dataset) - 1
        length = end - start
        x = self.data[start:end,:]
        x = x.transpose(1, 0)
        tgt = self.data[start+1:end+1, :]
        a = torch.ones((length, length), dtype=bool)
        mask = torch.triu(a, diagonal=0).transpose(1,0) #true means that we can see the token at this point

        return x, tgt, mask

def main():
    dataset = [
        [1, 2, 3],
        [1, 2],
        [1, 2, 3, 4],
        [1, 2],
        [1, 2, 3, 4, 5]
    ]
    batch_size = 4
    seq_len = 3
    dataloader = Dataloader(dataset, batch_size, seq_len)
    print(dataloader.data)
    # x, tgt, mask = dataloader.get_batch() #4X3 x, tgt; 3X3 mask
    x, tgt, mask = dataloader.get_batch(0)
    print(x, "\n", tgt, "\n", mask)

if __name__ == "__main__":
    main()
