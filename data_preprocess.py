import torch
import os
import numpy as np
import re
import underthesea
from transformers import AutoModel, AutoTokenizer

def load_bert():
    v_phobert = AutoModel.from_pretrained('vinai/phobert-base')
    v_tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base', use_fast=False)
    return v_phobert, v_tokenizer



def standardize_data(row):
    # xóa dấu chấm, phẩy, hỏi cuối câu
    row = re.sub(r"[\.,\?]+$-", "", row)
    # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
    row = row.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("-", " ").replace("?", " ")
    row = row.strip().lower()
    return row

def load_stopwords():
    sw = []
    with open("stopwords.txt", encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        sw.append(line.replace("\n",""))
    return sw


def make_bert_features(v_text):
    global phobert, sw
    v_tokenized = []
    max_len = 100 # Mỗi câu dài tối đa 100 từ
    for i_text in v_text:
        print("Đang xử lý line = ", i_text)
        # Phân thành từng từ
        line = underthesea.word_tokenize(i_text)
        # Lọc các từ vô nghĩa
        filtered_sentence = [w for w in line if not w in sw]
        # Ghép lại thành câu như cũ sau khi lọc
        line = " ".join(filtered_sentence)
        line = underthesea.word_tokenize(line, format="text")
        print(line)
        # print("Word segment  = ", line)
        # Tokenize bởi BERT
        line = tokenizer.encode(line)
        v_tokenized.append(line)

    # Chèn thêm số 1 vào cuối câu nếu như không đủ 100 từ
    padded = np.array([i + [1] * (max_len - len(i)) for i in v_tokenized])

    # Đánh dấu các từ thêm vào = 0 để không tính vào quá trình lấy features
    attention_mask = np.where(padded == 1, 0, 1)

    # Chuyển thành tensor
    padded = torch.tensor(padded).to(torch.long)
    attention_mask = torch.tensor(attention_mask)

    # Lấy features dầu ra từ BERT
    with torch.no_grad():
        last_hidden_states = phobert(input_ids= padded, attention_mask=attention_mask)

    v_features = last_hidden_states[0][:, 0, :].numpy()
    return v_features


if __name__ == '__main__':

    print("Chuẩn bị nạp danh sách các từ vô nghĩa (stopwords)...")
    sw = load_stopwords()
    print("Đã nạp xong danh sách các từ vô nghĩa")

    print("Chuẩn bị nạp model BERT....")
    phobert, tokenizer = load_bert()
    print("Đã nạp xong model BERT.")
    texts = ["Chúng tôi là những nghiên cứu viên"]
    print("Chuẩn bị tạo features từ BERT.....")
    features = make_bert_features(texts)
    print("Đã tạo xong features từ BERT")
    import pprint
    pprint.pprint(features)