import random
from torch.utils.data import Dataset, DataLoader
import itertools
import torch
import pandas as pd
import ast

class BERTDataset_For_League(Dataset):
    def __init__(self, data_pair, seq_len=13):
        
        self.seq_len = seq_len
        self.corpus_lines = len(data_pair)
        self.lines = data_pair

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):

        # Step 1: get random sentence pair, either negative or positive (saved as is_next_label)
        t1, t2, winner_label = self.get_sent(item)

        # Step 2: replace random words in sentence with mask / random words
        t1_random, t1_label = self.masking_word(t1)
        t2_random, t2_label = self.masking_word(t2)

        # Step 3: Adding CLS(1) and SEP(2) tokens to the start and end of sentences (Don't need padding since all input have the same size)
        t1 = [1] + t1_random + [2]
        t2 = t2_random + [2]
        
        # Step 4: combine sentence 1 and 2 as one input
        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = ([1] + t1_label + [2] + t2_label + [2])[:self.seq_len]
        
        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "winner_label": winner_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    def masking_word(self, sentence):
        tokens = sentence
        output_label = []
        output = []

        # 15% of the tokens would be replaced
        for i, token in enumerate(tokens):
            prob = random.random()
            token_id = [int(token)]
            
            if prob < 0.15:
                prob /= 0.15

                # 80% chance change token to mask(3) token
                if prob < 0.9:
                    for i in range(len(token_id)):
                        output.append(3)

                # 10% chance change token to original one
                # elif prob < 0.9:
                #     for i in range(len(token_id)):
                #         output.append(random.randrange(167)+4)

                else:
                    output.append(token_id)

                output_label.append(token_id)

            else:
                output.append(token_id)
                for i in range(len(token_id)):
                    output_label.append(0)

        # flattening
        output = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output]))
        output_label = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output_label]))
        assert len(output) == len(output_label)
        return output, output_label

    def get_sent(self, index):
        '''return random sentence pair'''
        t1, t2, label = self.get_corpus_line(index)
        return t1, t2, label

    def get_corpus_line(self, item):
        '''return sentence pair'''
        return self.lines[item][0], self.lines[item][1], self.lines[item][2]

class MLPDataset_For_League(Dataset):
    def __init__(self, data_pair, seq_len=13):
        
        self.seq_len = seq_len
        self.corpus_lines = len(data_pair)
        self.lines = data_pair

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):

        # Step 1: get random sentence pair, either negative or positive (saved as is_next_label)
        t1, t2, winner_label = self.get_sent(item)

        # Step 3: Adding CLS(1) and SEP(2) tokens to the start and end of sentences (Don't need padding since all input have the same size)
        t1 = [1] + t1 + [2]
        t2 = t2 + [2]
        
        # Step 4: combine sentence 1 and 2 as one input
        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        
        output = {"bert_input": bert_input,
                  "segment_label": segment_label,
                  "winner_label": winner_label}

        return {key: torch.tensor(value) for key, value in output.items()}


    def get_sent(self, index):
        '''return random sentence pair'''
        t1, t2, label = self.get_corpus_line(index)
        return t1, t2, label

    def get_corpus_line(self, item):
        '''return sentence pair'''
        return self.lines[item][0], self.lines[item][1], self.lines[item][2]


def make_champ_idx(file_path, start_idx = 4):
    df = pd.read_csv(file_path)
    df["teams"] = df["teams"].apply(lambda x : ast.literal_eval(x))

    # make train_datas
    # train_datas = [[df["teams"][i][:5], df["teams"][i][5:], df["winner"][i]] for i in range(len(df))]


    temp = []
    # concatenate all train_datas into one list, so we can make a set with that.
    for i in df["teams"]: 
        temp = temp+i

    temp_s  = sorted(list(set(temp)))

    champ_idx = {id: i+start_idx for i, id in enumerate(temp_s)}

    with open("champ_idx.txt", 'w') as file:
        for pair in champ_idx.items():
            file.write(f"{pair} \n")

    return champ_idx