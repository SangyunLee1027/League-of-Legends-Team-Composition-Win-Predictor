import pandas as pd
import ast
import torch
from Model.data_process import BERTDataset_For_League
from torch.utils.data import DataLoader
from Model.BERT import BERT,BERTLM, BERTTrainer



if __name__ == '__main__':
    df = pd.read_csv("Data/match_data_3.csv")
    df["teams"] = df["teams"].apply(lambda x : ast.literal_eval(x))


    champ_idx = {}
    with open("Model/champ_idx.txt", 'r') as file:
        for line in file:
            pair = ast.literal_eval(line)
            champ_idx.update({pair[0]: pair[1]})



    for i in range(len(df["teams"])):
        for j in range(len(df["teams"][i])):
            df["teams"][i][j] = champ_idx[df["teams"][i][j]]


    train_datas = [[df["teams"][i][:5], df["teams"][i][5:], df["winner"][i]] for i in range(len(df))]


    MAX_LEN = 33
    vocab_size = 3000

    train_data = BERTDataset_For_League(
    train_datas, seq_len=MAX_LEN)

    # print(train_data.)

    train_loader = DataLoader(
    train_data, batch_size=16, shuffle=True, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bert_model = BERT(
    vocab_size=vocab_size,
    d_model= 64,
    n_layers=8,
    heads=8,
    dropout=0.1,
    device = device
    )

    model = BERTLM(bert_model, vocab_size).to(device)
    # model.load_state_dict(torch.load("bert_model_final"))

    bert_trainer = BERTTrainer(model, train_loader, device=device)

    prev_epochs = 0
    epochs = 20


    for epoch in range(prev_epochs, epochs):
        bert_trainer.train(epoch)
        torch.save(model.state_dict(), "Trained_Model/bert_model_5")
    

