from flask import Flask, render_template, request
import json
from bilstm_crf import BiLSTM_CRF
import torch
import torch.nn as nn
from tqdm import tqdm
import re

app = Flask(__name__)


def sequence_padding(X, word2id, labels, max_len):
    input_ids_list, attention_mask_list = [], []
    pred_mask_list = []

    for i in tqdm(range(len(X))):
        # 获取句子X[i]中单词的id
        input_ids = [word2id.get(char, word2id["[UNK]"]) for char in X[i]]
        pred_mask = [1] * len(input_ids)
        attention_mask = [1] * len(input_ids)

        # 将不足max_len的input_id填充[PAD], labels填充labels.index("O")
        input_ids = input_ids[:max_len] + [word2id["[PAD]"]] * (max_len - len(input_ids))
        pred_mask = pred_mask[:max_len] + [0] * (max_len - len(pred_mask))
        attention_mask = attention_mask[:max_len] + [0] * (max_len - len(attention_mask))

        input_ids_list.append(input_ids)
        pred_mask_list.append(pred_mask)
        attention_mask_list.append(attention_mask)

    return torch.LongTensor(input_ids_list), torch.ByteTensor(attention_mask_list), torch.ByteTensor(pred_mask_list)


def predict(sentence):
    labels = ['O', 'B-LOC', 'B-ORG', 'B-T', 'I-LOC', 'I-PER', 'B-PER', 'I-ORG', 'I-T']
    max_len = 256
    num_labels = len(labels)
    embedding_dim = 50
    hidden_dim = 200
    hidden_dropout_prob = 0.2
    word2id = json.load(open('./data/word2id.json', 'r', encoding='utf-8'))
    model = BiLSTM_CRF(len(word2id), embedding_dim, hidden_dim, num_labels, hidden_dropout_prob)

    model.load_state_dict(torch.load('./save_model/bilstm.pt', map_location=torch.device("cpu")))

    sentence = sentence.replace('。', ' ')
    X = re.split(r'[。 .  ]',sentence)
    input_ids, attention_mask, pred_mask = sequence_padding(X, word2id, labels, max_len)
    predicted = model(
        input_ids=input_ids,
        attention_mask = attention_mask,
        pred_mask = pred_mask
    )
    pred = []
    predicted = predicted[0]
    predicted = [seq[seq>=0].tolist() for seq in predicted]
    for pl in predicted:
        pred.append([labels[l] for l in pl])
    return X, pred


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=["POST", "GET"])
def predict_result():
    if request.method == "POST":
        sent_dict = request.form.to_dict()
        sentences, predictions = predict(sent_dict['sentence'])
        res = [(sentences[_], predictions[_]) for _ in range(len(sentences))]
        return render_template("result.html", result=res)

if __name__ == "__main__":
    app.run()