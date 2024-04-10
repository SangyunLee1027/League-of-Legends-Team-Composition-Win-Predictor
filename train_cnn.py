import pandas as pd
import ast
from Model.data_process import MLPDataset_For_League
from torch.utils.data import DataLoader
from Model.BERT import BERT,BERTLM
from Model.CNN import CNN, CNN_Trainer
import torch

if __name__ == '__main__':


    df = pd.read_csv("Data/match_data_2.csv")
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

    MAX_LEN = 13
    vocab_size = 171

    train_data = MLPDataset_For_League(
    train_datas, seq_len=MAX_LEN)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(
    train_data, batch_size=16, shuffle=True, pin_memory=True)


    df2 = pd.read_csv("Data/match_data_validation.csv")
    df2["teams"] = df2["teams"].apply(lambda x : ast.literal_eval(x))



    for i in range(len(df2["teams"])):
        for j in range(len(df2["teams"][i])):
            df2["teams"][i][j] = champ_idx[df2["teams"][i][j]]


    test_dataset = [[df2["teams"][i][:5], df2["teams"][i][5:], df2["winner"][i]] for i in range(len(df2))]


    test_data = MLPDataset_For_League(
    test_dataset, seq_len=MAX_LEN)

    test_datas = {"bert_input": [], "segment_label": [], "winner_label": []}
    for data in test_data:
        for idx, d in data.items():
            test_datas[idx].append(d)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_datas = {key: torch.stack(value, dim = 0).to(device) for key, value in test_datas.items()}


    bert_model = BERT(
    vocab_size=vocab_size,
    d_model= 32,
    n_layers=8,
    heads=8,
    dropout=0.1,
    device = device
    )

    bert_ = BERTLM(bert_model, vocab_size)
    bert_.load_state_dict(torch.load("Trained_Model/bert_model_final"))


    cn = CNN(bert_).to(device)
    cn_trainer = CNN_Trainer(cn, train_loader, lr= 0.001, device='cuda')
    # cn.load_state_dict(torch.load("Trained_Model/cnn_model4"))

    prev_test_acc = 0

    prev_epochs = 0
    epochs = 20
    for epoch in range(prev_epochs, epochs):
        cn_trainer.train(epoch)
        
        cn.eval()

        winner_output = cn.forward(test_datas["bert_input"], test_datas["segment_label"])

        correct = torch.round(torch.flatten(winner_output)).eq(test_datas["winner_label"]).sum().item()
        test_acc = correct/len(winner_output)

        if prev_test_acc <= test_acc:
            print(f"saved at epoch: {epoch}")
            print(f"cur_acc: {test_acc}")
            prev_test_acc = test_acc
            torch.save(cn.state_dict(), "Trained_Model/cnn_model5")
    
    print(f"final accuracy: {prev_test_acc}")
