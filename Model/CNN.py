import torch.nn as nn
import torch
from torch import optim
import tqdm


class CNN(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels= 5, kernel_size = (2, 2), padding = 1)
        self.pool1 = nn.MaxPool2d(kernel_size = (1, 2))
        self.conv2 = nn.Conv2d(in_channels = 5, out_channels = 10, kernel_size = (5, 2), dilation = (2, 1))
        # self.pool2 = nn.MaxPool2d(kernel_size = (2, 1))
        self.relu = nn.ReLU()
        self.linear = nn.Linear(10 * 3 * 15, 2)

        self.bert = bert

        self.bert.eval()
        for param in self.bert.parameters():
            param.requires_grad = False


    def forward(self, x, segment_label):
        # get word embedding based on the champ Id
        embedded_x = self.bert.embedding(x, segment_label)
        
        # delete [SEP] tokens from input
        embedded_x = torch.cat((embedded_x[:, 1:6], embedded_x[:, 7:-1]), dim = 1)
        embedded_x = embedded_x.unsqueeze(1)

        # print()
        x = self.pool1(self.relu(self.conv1(embedded_x)))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        output = self.linear(x)
        return output
    



class CNN_Trainer:
    def __init__(
        self, 
        model, 
        train_dataloader, 
        test_dataloader=None, 
        lr= 1e-4,
        device='cuda'
        ):

        self.model = model
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr = lr)

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)


    def iteration(self, epoch, data_loader, train = True):
        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        
        mode = "train" if train else "test"

        # progress bar
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc="EP_%s:%d" % (mode, epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}"
        )

        for i, data in data_iter:

            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}
            
            # 1. forward the input data to get output
            winner_output = self.model.forward(data["bert_input"], data["segment_label"])
            
            # print(winner_output.shape)
            # 2-1. Crossentroyp loss of winner classification result
            loss = self.criterion(winner_output, (data["winner_label"]))

            # 3. backward and optimization only in train
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # next sentence prediction accuracy
            correct = winner_output.argmax(dim=-1).eq(data["winner_label"]).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += data["winner_label"].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

            if i % 10 == 0:
                data_iter.write(str(post_fix))
        print(
            f"EP{epoch}, {mode}: \
            avg_loss={avg_loss / len(data_iter)}, \
            total_acc={total_correct * 100.0 / total_element}"
        ) 

