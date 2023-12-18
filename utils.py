import os
import av
import json
import torch
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from seaborn import heatmap
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.cuda import FloatTensor, LongTensor
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from moviepy.editor import AudioFileClip, VideoFileClip
from transformers import AutoTokenizer, AutoProcessor, AutoModel
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, mean_absolute_error, matthews_corrcoef

def format_CH_SIMS(format_path: str):

    raw_label = pd.read_csv('./CH-SIMS/label.csv', names=['video','id','text','M','T','A','V','label','mode'])
    
    for mode in ["train", "valid", "test"]:
    
        labels = []
        path = f"{format_path}/{mode}"
        label = raw_label[raw_label['mode']==mode]
        os.makedirs(path, exist_ok=True)

        for i in label.index:
            if not os.path.isdir(f"{path}/{label.loc[i]['video']}"): 
                os.mkdir(f"{path}/{label.loc[i]['video']}")
            
            index = f"{label.loc[i]['video']}/{label.loc[i]['id']:0>4d}"
            labels += [{
                "id": index,
                "C":  label.loc[i]['label'],
                "R": label.loc[i]['M']
            }]
            
            with open(f"{path}/{index}.txt", 'w', encoding="utf-8") as f:
                f.write(label.loc[i]['text'])
                
            audio = AudioFileClip(f"./CH-SIMS/Raw/{index}.mp4")
            audio.write_audiofile(f"{path}/{index}.wav")
            
            video = VideoFileClip(f"./CH-SIMS/Raw/{index}.mp4")
            video.write_videofile(f"{path}/{index}.mp4")

        with open(f"{path}/label.json", 'w', encoding="utf-8") as file:
            json.dump(labels, file, indent=4)


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    
    return indices


def read_video_pyav(container, indices):

    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    data = np.stack([x.to_ndarray(format="rgb24") for x in frames])
    
    return data


def Text(path, tokenizer, model):
    
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    with open(f"{path}.txt", 'r', encoding='UTF-8') as f:
        data = f.read()
    inputs = tokenizer(data, max_length=49, padding='max_length', return_tensors="pt").to(device)
    with torch.no_grad():
        data = model(**inputs).last_hidden_state
    data = data.cpu()
    
    return data


def Audio(path, processor, model):
    
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    waveform, sample_rate = librosa.load(f"{path}.wav", sr=16000)
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt").to(device)
    with torch.no_grad():
        data = model(**inputs).last_hidden_state
    data = data.cpu()
    
    return data


def Vision(path, imageprocessor, model):
    
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    container = av.open(f"{path}.mp4")
    indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
    video = read_video_pyav(container, indices)
            
    inputs = imageprocessor(videos=list(video), return_tensors="pt").to(device)
    with torch.no_grad():
        data = model.get_video_features(**inputs)
    data = data[None, :].cpu()
    
    return data


def extracted_features(
        format_path: str,
        text_model: str=None,
        audio_model: str=None,
        vision_model: str=None,
        data_save_to: str=None,
        ) -> dict():

    format_CH_SIMS(format_path)
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    encoder = {"Negative":0, "Neutral":1, "Positive":2}
    data = {}
    
    if text_model:
        tokenizer = AutoTokenizer.from_pretrained(text_model)
        model_text = AutoModel.from_pretrained(text_model).to(device)
    if audio_model:
        processor_audio = AutoProcessor.from_pretrained(audio_model)
        model_audio = AutoModel.from_pretrained(audio_model).to(device)
    if vision_model:
        processor_vision = AutoProcessor.from_pretrained(vision_model)
        model_vision = AutoModel.from_pretrained(vision_model).to(device)
    
    for mode in os.listdir(format_path):

        data[mode] = {}
        with open(f"{format_path}/{mode}/label.json", "r", encoding="utf-8") as file:
            label = json.load(file)
        first = True
        
        for i in tqdm(range(len(label)), desc=f"{mode}", unit_scale=True, colour="green"):
            path = f"{format_path}/{mode}/{label[i]['id']}"

            if first :
                first = False
                if text_model: 
                    data_text = Text(path, tokenizer, model_text)
                if audio_model: 
                    data_audio = Audio(path, processor_audio, model_audio)
                if vision_model: 
                    data_vision = Vision(path, processor_vision, model_vision)
                label_c = np.array([encoder[label[i]["C"]]])
                label_r = np.array([label[i]["R"]])
            else:
                if text_model: 
                    data_text = np.append(data_text, Text(path, tokenizer, model_text), axis=0)
                if audio_model: 
                    data_audio = np.append(data_audio, Audio(path, processor_audio, model_audio), axis=0)
                if vision_model: 
                    data_vision = np.append(data_vision, Vision(path, processor_vision, model_vision), axis=0)
                label_c = np.append(label_c, encoder[label[i]["C"]])
                label_r = np.append(label_r, label[i]["R"])
        if text_model: 
            data[mode]["text"] = data_text
        if audio_model: 
            data[mode]["audio"] = data_audio
        if vision_model: 
            data[mode]["vision"] = data_vision
        data[mode]["label_c"] = label_c
        data[mode]["label_r"] = label_r

    if data_save_to:
        torch.save(data, data_save_to, pickle_protocol=4)

    return data

class dataset(Dataset):
    def __init__(self, data, mode):
        
        self.text = FloatTensor(data[mode]["text"])
        self.audio = FloatTensor(data[mode]["audio"])
        self.vision = FloatTensor(data[mode]["vision"])
        self.label_r = FloatTensor(data[mode]["label_r"])
        self.label_c = LongTensor(data[mode]["label_c"])

    def __len__(self):
        return len(self.label_c)
    
    def __getitem__(self, index):

        data = {
            'text': self.text[index],
            'audio': self.audio[index],
            'vision': self.vision[index],
        }
        label = {
            "c": self.label_c[index],
            "r": self.label_r[index],
        }
        
        return data, label


def split_data(data, batch_size):
    
    train_set = dataset(data, mode="train")
    valid_set = dataset(data, mode="valid")
    test_set = dataset(data, mode="test")
    
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size)
    valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size)
    
    return train_loader, valid_loader, test_loader


class EarlyStopping:

    def __init__(self, save_path, patience=10, verbose=True):

        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = -1
        self.early_stop = False

        os.makedirs(save_path, exist_ok=True)

    def __call__(self, score, model):

        if score >= self.best_score:
            self.save_checkpoint(score, model)
            self.best_score = score
            self.counter = 0

        else :
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping!")

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation accuracy improvement ({self.best_score*100:.2f} --> {score*100:.2f}).  Saving model ...')
        torch.save(model, self.save_path)	

def validation(model, loss_function, dataloader, regression):
    
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    model = model.to(device)
    
    running_loss = 0
    all_pred = torch.tensor([]).to(device)
    all_true = torch.tensor([]).to(device)

    with torch.no_grad():
        
        model.eval()
        
        for data, label in tqdm(dataloader, desc="Valid", ncols=100, unit_scale=True, colour="red"):
            
            outputs = model(data["text"], data["audio"], data["vision"])
            label =  label["r" if regression else "c"]

            loss = loss_function(outputs, label)
            running_loss += loss.item()
            
            predicted = outputs if regression else c

            all_pred = torch.cat((all_pred, predicted), dim=0)
            all_true = torch.cat((all_true, label), dim=0)
            
            torch.cuda.empty_cache()
            
    loss = running_loss / len(dataloader)

    return loss, all_true.cpu(), all_pred.cpu().squeeze()


def Train(model, loss_function, train_loader, valid_loader, lr, eposhs, early_stop, model_save_to, regression):
    
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(model_save_to, patience=early_stop)
    
    history = {}
    history["train loss"] = []
    history["train em"] = [] 
    history["valid loss"] = []
    history["valid em"] = []

    for epoch in range(eposhs):
        
        model.train()
        running_loss = 0.0
        true = torch.tensor([]).to(device)
        pred = torch.tensor([]).to(device)
        
        for data, label in tqdm(train_loader, desc=f"Epoch {epoch+1}", unit_scale=True, colour="blue"):
           
            optimizer.zero_grad(set_to_none=True)
            
            outputs = model(data["text"], data["audio"], data["vision"])
            label =  label["r" if regression else "c"]    
            loss = loss_function(outputs, label)
            
            predicted = outputs if regression else torch.argmax(outputs, axis=1)
            true = torch.cat((true, label), dim=0)
            pred = torch.cat((pred, predicted), dim=0)
            

            running_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        
        true = true.detach().cpu()
        pred = pred.detach().cpu().squeeze()
        train_loss = running_loss / len(train_loader)
        train_em = np.corrcoef(pred, true)[0][1] if regression else accuracy_score(true, pred)
        valid_loss, true, pred = validation(model, loss_function, valid_loader, regression)
        valid_em = np.corrcoef(pred, true)[0][1] if regression else accuracy_score(true, pred)


        print(f"Train Loss: {train_loss:.4f}\t" \
              f"Train EM: {train_em*100:.2f}\t\t" \
              f"Valid Loss: {valid_loss:.4f}\t" \
              f"Valid EM: {valid_em*100:.2f}\t" \
              )
                
        history["train loss"].append(train_loss)
        history["train em"].append(train_em)
        history["valid loss"].append(valid_loss)
        history["valid em"].append(valid_em)

        early_stopping(valid_em, model)
        if early_stopping.early_stop: 
            break

    return model, history

def visualization(history, true, pred, regression):
    
    print("="*100)
    if regression:
        mae = mean_absolute_error(true, pred)
        corr = np.corrcoef(pred, true)[0][1]
        print(f"MAE: {mae*100:.2f}")
        print(f"CORR: {corr*100:.2f}")

    else:
        acc = accuracy_score(true, pred)
        f1 = f1_score(true, pred, average='weighted')
        print(f"Accuracy: {acc:.2%}")
        print(f"F1 Score: {f1:.2%}")
        
        target_names = ["Negative", "Neutral", "Positive"]
        print (classification_report(true, pred, target_names=target_names))

    plt.figure()
    plt.plot(history["train loss"])
    plt.plot(history["valid loss"])
    plt.legend(["Train", "Valid"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Learning Curve Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"learning_curves_loss")

    plt.figure()
    plt.plot(history["train em"])
    plt.plot(history["valid em"])
    plt.legend(["Train", "Valid"])
    plt.xlabel("Epochs")
    plt.ylabel("Evaluation Matrix")
    plt.title(f"Learning Curve Evaluation Matrix")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"learning_curves_em")
