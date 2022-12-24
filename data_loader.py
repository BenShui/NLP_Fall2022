import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def read_file(filename):
    """
    :param filename: 文件地址
    :return: X, y: X是包含所有句子的list，一个句子占据一个list；y是对应的标签
    样例输入：data/small_train.txt
    样例输出：X: [['中', '国', '很', '大'], ['句', '子', '结', '束', '是', '空', '行']],
    y: [['B-LOC', 'I-LOC', 'O', 'O'], ['O', 'O', 'O', 'O', 'O', 'O', 'O']]
    """
    X, y = [], []
    labels = []
    with open(filename, "r", encoding='utf-8') as f:
        x0, y0 = [], []
        for line in f:
            data = line.strip()
            if data:
                x0.append(data.split()[0])
                y0.append(data.split()[1])
            else:
                if len(x0) != 0:
                    X.append(x0)
                    y.append(y0)
                x0, y0 = [], []
        if len(x0) != 0:
            X.append(x0)
            y.append(y0)
    return X, y


def sequence_padding_bilstm(X, y, word2id, labels, max_len):
    """
    :param X: 文本
    :param y: 标签
    :param word2id: word2id 词典
    :param labels: labels = ['O', 'B-LOC', 'B-ORG', 'B-T', 'I-LOC', 'I-PER', 'B-PER', 'I-ORG', 'I-T']
    :param max_len: 最大句子长度
    :return:
    """
    input_ids_list = []
    attention_mask_list = []
    pred_mask_list = []
    input_labels_list = []

    for i in tqdm(range(len(X))):
        # 获取句子X[i]中单词的id
        input_ids = [word2id.get(char, word2id["[UNK]"]) for char in X[i]]
        input_labels = [labels.index(l) for l in y[i]]
        pred_mask = [1] * len(input_ids)
        attention_mask = [1] * len(input_ids)

        # 将不足max_len的input_id填充[PAD], labels填充labels.index("O")
        input_ids = input_ids[:max_len] + [word2id["[PAD]"]] * (max_len - len(input_ids))
        input_labels = input_labels[:max_len] + [labels.index("O")] * (max_len - len(input_labels))
        pred_mask = pred_mask[:max_len] + [0] * (max_len - len(pred_mask))
        attention_mask = attention_mask[:max_len] + [0] * (max_len - len(attention_mask))

        input_ids_list.append(input_ids)
        input_labels_list.append(input_labels)
        pred_mask_list.append(pred_mask)
        attention_mask_list.append(attention_mask)

    return torch.LongTensor(input_ids_list), torch.ByteTensor(attention_mask_list), torch.ByteTensor(pred_mask_list), torch.LongTensor(input_labels_list)


class NERDataset(Dataset):
    def __init__(self, file_path, labels, word2id=None, max_len=128):
        self.X, self.y = read_file(file_path)
        self.input_ids, self.attention_masks, self.pred_mask, self.input_labels = sequence_padding_bilstm(self.X,
                                                                                                          self.y,
                                                                                                          word2id=word2id,
                                                                                                          labels=labels,
                                                                                                          max_len=max_len)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.pred_mask[idx], self.input_labels[idx]


# if __name__ == "__main__":
#     labels = ['O', 'B-LOC', 'B-ORG', 'B-T', 'I-LOC', 'I-PER', 'B-PER', 'I-ORG', 'I-T']
#     word2id = {'[PAD]': 0, '[UNK]': 1, '中': 2, '国': 3, '很': 4,
#                '大': 5, '句': 6, '子': 7, '结': 8, '束': 9, '是': 10, '空': 11, '行': 12}
#     test_ner = NERDataset('data/small_train.txt', labels, word2id, max_len=10)
#     print(test_ner[1])
#     print(test_ner[0])