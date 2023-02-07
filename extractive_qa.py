#!/usr/bin/env python
# coding: utf-8

# ---
# # Read files

# In[ ]:


import torch
import pandas as pd
import numpy as np
from more_itertools import sliced
import re

# Train Data

def read_data(data):
    with open(data, 'r') as f:
        lines = f.readlines()

    bismillah = []

    for i in range(len(lines)):
        bismillah.append(lines[i].split('|||'))

    df = pd.DataFrame(bismillah, columns=['snippets', 'question', 'answer']) # Create dataframe
    df = df.replace('\n', '', regex=True) 
    df = df.dropna()

    # splitting the snippets based on </s> tag
    df = df.set_index(['question', 'answer']).apply(lambda x: x.str.split('</s>').explode()).reset_index()
    # delete <s> tag
    df['snippets'] = df['snippets'].str.replace('<s>', '') 
    # delete unnecessary whitespace
    df['snippets'] = df['snippets'].str.strip() 
    df['question'] = df['question'].str.strip()
    df['answer'] = df['answer'].str.strip()
    # drop empty snippets
    df.drop(df[df['snippets']==""].index, inplace=True)
    df = df.reset_index(drop=True)
    
    return df

df_train = read_data('train.txt')
df_val = read_data('val.txt')
df_test = read_data('test.txt')


# ---
# # Preprocessing

# In[ ]:


# add (start, end) index
def index_of_substring(main_string, substring):
    try:
        start_index = main_string.index(substring)
        end_index = start_index + len(substring) -1
        return(pd.Series([start_index,end_index]))
    except ValueError:
        return(pd.Series([None,None]))
    
df_train[["start","end"]] = df_train.apply(lambda row:index_of_substring(row['snippets'],row["answer"]),axis=1)
df_val[["start","end"]] = df_val.apply(lambda row:index_of_substring(row['snippets'],row["answer"]),axis=1)


# In[ ]:


# Change '' into NaN
df_train['snippets'] = df_train['snippets'].replace('', np.nan)
df_train['question'] = df_train['question'].replace('', np.nan)
df_val['snippets'] = df_val['snippets'].replace('', np.nan)
df_val['question'] = df_val['question'].replace('', np.nan)
df_test['snippets'] = df_test['snippets'].replace('', np.nan)
df_test['question'] = df_test['question'].replace('', np.nan)

# Take the not empty instance only 
df_train = df_train[df_train['snippets'].notna()]
df_train = df_train[df_train['question'].notna()]
df_val = df_val[df_val['snippets'].notna()]
df_val = df_val[df_val['question'].notna()]
df_test = df_test[df_test['snippets'].notna()]
df_test = df_test[df_test['question'].notna()]


# In[ ]:


# Add yn column to know snippets contains answer or not
df_train['yn'] = ((df_train['start'].notna()))
df_val['yn'] = ((df_val['start'].notna()))


# In[ ]:


import timeit
start = timeit.default_timer()

# Sampling 1 answered question and 1 unanswered question for each question
# df_train_1 = df_train.groupby(['question', 'yn']).dropna()
# df_val_1 = df_val.groupby(['question', 'yn']).dropna()
df_train_2 = df_train.groupby(['question', 'yn']).sample(n=4, random_state=11, replace=True).reset_index(drop=True)
df_val_2 = df_val.groupby(['question', 'yn']).sample(n=4, random_state=11, replace=True).reset_index(drop=True) 

stop = timeit.default_timer()

print('Time: ', stop - start)  


# In[ ]:


df_train_2 = df_train_2.drop_duplicates(subset=['snippets'])
df_val_2 = df_val_2.drop_duplicates(subset=['snippets'])
df_val_2 = df_val_2.dropna()

# Fill Nan in unanswered question-snippets pair
df_train_2.fillna(0, inplace=True)
df_val_2.fillna(0, inplace=True)


# In[ ]:


df_train_2.to_csv('train_sample_4.csv', index=False, header=False)
df_val_2.to_csv('val_sample_4.csv', index=False, header=False)


# ---
# # Retrieval for the test data with BM25

# In[ ]:


from rank_bm25 import BM25Okapi
import string


# In[ ]:


corpus = df_test['snippets'].tolist()      # create corpus

new_corpus = []
for i in corpus:
    i = i.translate(str.maketrans('', '', string.punctuation))      # remove punctuation
    i = i.strip()       # remove extra whitespace in front and behind snippet
    i = i.replace("  ", "")      # remove double whitespace
    new_corpus.append(i)


# In[ ]:


from collections import deque
from tqdm import tqdm

x = [doc.split(" ") for doc in new_corpus]      # create corpus for bm25
bm25 = BM25Okapi(x)      # init bm25

question = deque()
snippets = deque()

for i, q in enumerate(tqdm(df_test['question'].unique())):
    x = q.split(" ")     # splitting
    result = bm25.get_top_n(x, new_corpus, n=20)     # get 20 most relevant snippets
    result = ' '.join(result)     # joining element of list result
    question.append(q)     # append to final list
    snippets.append(result)     # append to final list
    # print(i)

lists = [question, snippets]
df = pd.concat([pd.Series(x) for x in lists], axis=1)
df.rename(columns={0: 'question', 1: 'snippets'}, inplace=True)
df.to_csv('test_bm25_top20.csv', index=False)


# ---
# # Finetuning Bert

# In[ ]:


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import pandas as pd
# import cudf
import numpy as np
from more_itertools import sliced
import re

cols = ['question', 'answer', 'snippets', 'start', 'end', 'yn']
df_train_1 = pd.read_csv('train_sample_4.csv', names=cols)
df_val_1 = pd.read_csv('val_sample_4.csv', names=cols)
df_test = pd.read_csv('test_bm25_top20.csv')


# In[ ]:


import transformers
transformers.logging.set_verbosity_error()
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(df_train_1.snippets.apply(str).tolist(), df_train_1.question.apply(str).tolist(), truncation=True, padding=True)
val_encodings = tokenizer(df_val_1.snippets.apply(str).tolist(), df_val_1.question.apply(str).tolist(), truncation=True, padding=True)
test_encodings = tokenizer(df_test.snippets.apply(str).tolist(), df_test.question.apply(str).tolist(), truncation=True, padding=True)


# In[ ]:


# REAL CODE FOR QA WITH START AND END POS  (BERT FOR QA)
def add_token_positions(encodings, df):
    start_positions = []
    end_positions = []
    for i in range(len(df)):
        start_positions.append(encodings.char_to_token(i, df.start[i].astype(int)))
        end_positions.append(encodings.char_to_token(i, df.end[i].astype(int)))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            # start_positions[-1] = tokenizer.model_max_length
            start_positions[-1] = 0
            end_positions[-1] = 0
        # end position cannot be found, char_to_token found space, so shift one token forward
        go_back = 1
        while end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, df.end[i].astype(int)-go_back)
            go_back += 1

    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

add_token_positions(train_encodings, df_train_1)
add_token_positions(val_encodings, df_val_1)


# In[ ]:


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = Dataset(train_encodings)
val_dataset = Dataset(val_encodings)
test_dataset = Dataset(test_encodings)


# In[ ]:


from transformers import BertModel
from torch import nn


class myModel(torch.nn.Module):

    def __init__(self):

        super(myModel, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Sequential(
            nn.Linear(768, 2))
            # nn.Dropout(0.8),
            # nn.Linear(512, 256),
            # nn.Dropout(0.9),
            # nn.Linear(256, 2))

    def forward(self, input_ids, attention_mask, token_type_ids):
    # def forward(self, input_ids, attention_mask):

        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        logits = output[0]
        out = self.fc(logits)

        return out


# In[ ]:


from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

# batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

torch_device = torch.device("cuda:2")
epochs = 2
total_steps = len(train_loader) * epochs
model = myModel().to(torch_device)
optim = AdamW(model.parameters(), lr=3e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=10000, num_training_steps=total_steps)
criterion = nn.CrossEntropyLoss()


# In[ ]:


def evaluate(valid_loader):
    acc = []
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        loop = tqdm(val_loader, leave=True)
        for batch_id, batch in enumerate(loop):
            input_ids = batch['input_ids'].to(torch_device)
            attention_mask = batch['attention_mask'].to(torch_device)
            token_type_ids = batch['token_type_ids'].to(torch_device)
            start_positions = batch['start_positions'].to(torch_device)
            end_positions = batch['end_positions'].to(torch_device)
            
            # Outputs
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            # outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            start_positions_logits, end_positions_logits = torch.split(outputs, 1, 2)
            
            start_pred = torch.argmax(start_positions_logits, dim=1)
            end_pred = torch.argmax(end_positions_logits, dim=1)
            
            start_positions_logits = start_positions_logits.squeeze(-1).contiguous()
            end_positions_logits = end_positions_logits.squeeze(-1).contiguous()
        
            start_positions_loss = criterion(start_positions_logits, start_positions)
            end_positions_loss = criterion(end_positions_logits, end_positions)
            
            loss = start_positions_loss + end_positions_loss
            
            start_positions = start_positions.reshape([start_positions.size(dim=0), 1])
            end_positions = end_positions.reshape([end_positions.size(dim=0), 1])
            start_score = ((start_pred == start_positions).sum()/len(start_pred)).item()
            end_score = ((end_pred == end_positions).sum()/len(end_pred)).item()
            
            acc.append(start_score)
            acc.append(end_score)
            
            running_loss += loss.item()
            if batch_id % 800 == 0 and batch_id != 0:
                print('Validation Epoch {} Batch {} Loss {:.4f}'.format(batch_id+1, batch_id, running_loss/50))
                running_loss = 0.0
        acc = sum(acc)/len(acc)
                    
        print('Evaluate loss: ', loss)
        print('Accuracy SAMPLED5-5 DROPPED SOME: ', acc)
        
# evaluate(val_loader)


# In[ ]:


for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    loop = tqdm(train_loader, leave=True)
    for batch_id, batch in enumerate(loop):
        # Train
        optim.zero_grad()
        
        input_ids = batch['input_ids'].to(torch_device)
        attention_mask = batch['attention_mask'].to(torch_device)
        token_type_ids = batch['token_type_ids'].to(torch_device)
        start_positions = batch['start_positions'].to(torch_device)
        end_positions = batch['end_positions'].to(torch_device)
        
        # Outputs
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        
        start_positions_logits, end_positions_logits = torch.split(outputs, 1, 2)
        
        start_positions_logits = start_positions_logits.squeeze(-1).contiguous()
        end_positions_logits = end_positions_logits.squeeze(-1).contiguous()
        
        start_positions_loss = criterion(start_positions_logits, start_positions)
        end_positions_loss = criterion(end_positions_logits, end_positions)
        
        loss = start_positions_loss + end_positions_loss
        
        # Calculateing loss
        loss.backward()
        # Update params
        optim.step()
        scheduler.step()
        
        running_loss += loss.item()
        if batch_id % 1600 == 0 and batch_id != 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(batch_id+1, batch_id, running_loss/50))
            running_loss = 0.0
            
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
    evaluate(val_loader)


# In[ ]:


# Save model

save_path = './finetuned-bert-base-4sampledroppedsome-2ep-searchqa'
torch.save(model.state_dict(), save_path)


# ---
# # Predict

# In[ ]:


save_path = './finetuned-bert-base-4sampledroppedsome-2ep-searchqa'
model.load_state_dict(torch.load(save_path))


# In[ ]:


from tqdm import tqdm

# Evaluation function
def predict(test_loader):
    predict_pos, sub_output = [], []
    model.eval()
    loop = tqdm(test_loader, leave=True)
    for batch_id, batch in enumerate(loop):
        input_ids = batch['input_ids'].to(torch_device)
        attention_mask = batch['attention_mask'].to(torch_device)
        token_type_ids = batch['token_type_ids'].to(torch_device)
            
        # Outputs
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)            
            
        start_positions_logits, end_positions_logits = torch.split(outputs, 1, 2)
        
        start_positions_logits = start_positions_logits.squeeze(-1).contiguous()
        end_positions_logits = end_positions_logits.squeeze(-1).contiguous()
        
        start_positions_preds = torch.argmax(start_positions_logits, 1).cpu().numpy()
        end_positions_preds = torch.argmax(end_positions_logits, 1).cpu().numpy()
            
        for i in range(len(input_ids)):
            predict_pos.append((start_positions_preds[i].item(), end_positions_preds[i].item()))
                
            sub = tokenizer.decode(input_ids[i][start_positions_preds[i]:end_positions_preds[i]+1])
                
            sub_output.append(sub)
                
    return sub_output, predict_pos


# In[ ]:


sub_output, predict_pos = predict(test_loader)


# In[ ]:


df_test.drop('snippets', axis=1, inplace=True)
df_test['answer'] = sub_output


# In[ ]:


import csv

df_test1 = df_test[['question', 'answer']]
df_final = df_test1.apply(lambda row: ' ||| '.join(row.values.astype(str)), axis=1) # joining column
df_final = df_final.replace(',','‚', regex=True) #for direct use of QUOTE_NONE without defining escape_char
df_final.to_csv('test_submit_infer.txt', index=False, header=False, quoting=csv.QUOTE_NONE)

