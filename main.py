## IMPORT LIBRARIES
import os
import pandas as pd
import numpy as np
from tqdm import tqdm, trange

import torch 
from torch import tensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertPreTrainedModel, BertModel
from transformers import AutoConfig, AutoTokenizer

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

## LOAD DATASETS
train_df = pd.read_csv('Emotion Dataset/train.txt', sep=';')
test_df = pd.read_csv('Emotion Dataset/test.txt', sep=';')
val_df = pd.read_csv('Emotion Dataset/val.txt', sep=';')

print(train_df.shape, test_df.shape, val_df.shape)

train_df.columns = ['sentence', 'emotion']
test_df.columns = ['sentence', 'emotion']
val_df.columns = ['sentence', 'emotion']

print(train_df.head())

## CONFIGS
MODEL_OUT_DIR = 'models/bert_emotion'
TRAIN_FILE_PATH = 'Emotion Dataset/train.txt'
VALID_FILE_PATH = 'Emotion Dataset/val.txt'
TEST_FILE_PATH = 'Emotion Dataset/test.txt'
## Model Configurations
MAX_LEN_TRAIN = 68
MAX_LEN_VALID = 68
MAX_LEN_TEST = 68
BATCH_SIZE = 160
LR = 1e-5
NUM_EPOCHS = 10
NUM_THREADS = 1  ## Number of threads for collecting dataset
MODEL_NAME = 'bert-base-uncased'
LABEL_DICT = {'joy':0, 'sadness':1, 'anger':2, 'fear':3, 'love':4, 'surprise':5}

if not os.path.isdir(MODEL_OUT_DIR):
    os.makedirs(MODEL_OUT_DIR)

## CREATE DATASET
class Emotions_Dataset(Dataset):

    def __init__(self, filename, maxlen, tokenizer, label_dict): 
        #Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename, delimiter = ';')
        # name columns
        self.df.columns = ['sentence', 'emotion']
        #Initialize the tokenizer for the desired transformer model
        self.df['emotion'] = self.df['emotion'].map(label_dict)
        self.tokenizer = tokenizer
        #Maximum length of the tokens list to keep all the sequences of fixed size
        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):    
        #Select the sentence and label at the specified index in the data frame
        sentence = self.df.loc[index, 'sentence']
        label = self.df.loc[index, 'emotion']
        #Preprocess the text to be suitable for the transformer
        tokens = self.tokenizer.tokenize(sentence) 
        tokens = ['[CLS]'] + tokens + ['[SEP]'] 
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))] 
        else:
            tokens = tokens[:self.maxlen-1] + ['[SEP]'] 
        #Obtain the indices of the tokens in the BERT Vocabulary
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens) 
        input_ids = torch.tensor(input_ids) 
        #Obtain the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attention_mask = (input_ids != 0).long()
        
        label = torch.tensor(label, dtype=torch.long)
        
        return input_ids, attention_mask, label

class BertEmotionClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        #The classification layer that takes the [CLS] representation and outputs the logit
        self.cls_layer = nn.Linear(config.hidden_size, 6)

    def forward(self, input_ids, attention_mask):
        #Feed the input to Bert model to obtain contextualized representations
        reps, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #Obtain the representations of [CLS] heads
        cls_reps = reps[:, 0]
        logits = self.cls_layer(cls_reps)
        return logits

## TRAINING FUNCTION
def train(model, criterion, optimizer, train_loader, val_loader, epochs, device):
    best_acc = 0
    for epoch in trange(epochs, desc="Epoch"):
        model.train()
        train_acc = 0
        for i, (input_ids, attention_mask, labels) in enumerate(iterable=train_loader):
            optimizer.zero_grad()  
            
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_acc += get_accuracy_from_logits(logits, labels)
        
        print("Training accuracy is {train_acc/len(train_loader)}")
        val_acc, val_loss = evaluate(model=model, criterion=criterion, dataloader=val_loader, device=device)
        print("Epoch {} complete! Validation Accuracy : {}, Validation Loss : {}".format(epoch, val_acc, val_loss))
        
#         if val_acc > best_acc:
#             print("Best validation accuracy improved from {} to {}, saving model...".format(best_acc, val_acc))
#             best_acc = val_acc
#             model.save_pretrained(save_directory=MODEL_OUT_DIR + '/')
#             config.save_pretrained(save_directory=MODEL_OUT_DIR + '/')
#             tokenizer.save_pretrained(save_directory=MODEL_OUT_DIR + '/')

## EVALUATION FUNCTION
def evaluate(model, criterion, dataloader, device):
    model.eval()
    mean_acc, mean_loss, count = 0, 0, 0
#     predicted_labels = []
#     actual_labels = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in (dataloader):
            
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            logits = model(input_ids, attention_mask)
            
            mean_loss += criterion(logits.squeeze(-1), labels).item()
            mean_acc += get_accuracy_from_logits(logits, labels)
            count += 1
            
#             predicted_labels += output
#             actual_labels += labels
            
    return mean_acc/count, mean_loss/count

def get_accuracy_from_logits(logits, labels):
    probs = F.softmax(logits, dim=1)
    output = torch.argmax(probs, dim=1)
    acc = (output == labels).float().mean()
    return acc

## PREDICT FUNCTION
def predict(model, dataloader, device):
    predicted_label = []
    actual_label = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in (dataloader):
            
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            logits = model(input_ids, attention_mask)
            
            probs = F.softmax(logits, dim=1)
            output = torch.argmax(probs, dim=1)
            
            predicted_label += output
            actual_label += labels
            
    return predicted_label, actual_label

## Configuration loaded from AutoConfig 
config = AutoConfig.from_pretrained(MODEL_NAME)
## Tokenizer loaded from AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
## Creating the model from the desired transformer model
model = BertEmotionClassifier.from_pretrained(MODEL_NAME, config=config)
## GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
## Putting model to device
model = model.to(device)
## Takes as the input the logits of the positive class and computes the binary cross-entropy 
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()
## Optimizer
optimizer = optim.Adam(params=model.parameters(), lr=LR)

## Training Dataset
train_set = Emotions_Dataset(filename=TRAIN_FILE_PATH, maxlen=MAX_LEN_TRAIN, tokenizer=tokenizer, label_dict=LABEL_DICT)
valid_set = Emotions_Dataset(filename=VALID_FILE_PATH, maxlen=MAX_LEN_VALID, tokenizer=tokenizer, label_dict=LABEL_DICT)
test_set = Emotions_Dataset(filename=TEST_FILE_PATH, maxlen=MAX_LEN_TEST, tokenizer=tokenizer, label_dict=LABEL_DICT)


## Data Loaders
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)
valid_loader = DataLoader(dataset=valid_set, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)
test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)

# print(len(train_loader))
if __name__ == '__main__':
    train(model=model, 
        criterion=criterion,
        optimizer=optimizer, 
        train_loader=train_loader,
        val_loader=valid_loader,
        epochs = 5,
        device = device)

    actual_label, predicted_label = predict(model, test_loader, device=device)
    actual_label = np.array([item.to('cpu') for item in actual_label])
    predicted_label = np.array([item.to('cpu') for item in predicted_label])

    print("Accuracy :",metrics.accuracy_score(actual_label, predicted_label))
    print("f1 score macro :",metrics.f1_score(actual_label, predicted_label, average = 'macro'))
    print("f1 scoore micro :",metrics.f1_score(actual_label, predicted_label, average = 'micro'))
    print("Hamming loss :",metrics.hamming_loss(actual_label, predicted_label))
    print("Classification Report: \n", classification_report(actual_label, predicted_label))
    print("Confusion Matrix: \n", confusion_matrix(actual_label, predicted_label))

