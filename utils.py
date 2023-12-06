import os
import av
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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

def format_CH_SIMS():

    raw_label = pd.read_csv('./CH-SIMS/label.csv', names=['video','id','text','M','T','A','V','label','mode'])
    
    for mode in ["train", "valid", "test"]:
    
        labels = {}
        path = f"./formatted/{mode}"
        label = raw_label[raw_label['mode']==mode]
        os.makedirs(path, exist_ok=True)

        for i in label.index:
            if not os.path.isdir(f"{path}/{label.loc[i]['video']}"): 
                os.mkdir(f"{path}/{label.loc[i]['video']}")
            
            index = f"{label.loc[i]['video']}/{label.loc[i]['id']:0>4d}"
            labels[index] = label.loc[i]['label']
            
            with open(f"{path}/{index}.txt", 'w', encoding='UTF-8') as f:
                f.write(label.loc[i]['text'])
                
            audio = AudioFileClip(f"./CH-SIMS/Raw/{index}.mp4")
            audio.write_audiofile(f"{path}/{index}.wav")
            
            video = VideoFileClip(f"./CH-SIMS/Raw/{index}.mp4")
            video.write_videofile(f"{path}/{index}.mp4")

        pd.DataFrame(list(labels.items())).to_csv(f"{path}/label.csv", index=False, header=False)


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
        data_path: str,
        text_model: str=None,
        audio_model: str=None,
        vision_model: str=None,
        data_save_to: str=None,
        ) -> dict():
    
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
    
    for mode in os.listdir(data_path):

        data[mode] = {}
        labels = pd.read_csv(f"{data_path}/{mode}/label.csv", names=["id", "label"])
        first = True
        
        for i in tqdm(range(len(labels)), desc=f"{mode}", unit_scale=True, colour="green"):
            path = f"{data_path}/{mode}/{labels['id'][i]}"

            if first :
                first = False
                if text_model: 
                    data_text = Text(path, tokenizer, model_text)
                if audio_model: 
                    data_audio = Audio(path, processor_audio, model_audio)
                if vision_model: 
                    data_vision = Vision(path, processor_vision, model_vision)
                label = np.array([encoder[labels["label"][i]]])
            else:
                if text_model: 
                    data_text = np.append(data_text, Text(path, tokenizer, model_text), axis=0)
                if audio_model: 
                    data_audio = np.append(data_audio, Audio(path, processor_audio, model_audio), axis=0)
                if vision_model: 
                    data_vision = np.append(data_vision, Vision(path, processor_vision, model_vision), axis=0)
                label = np.append(label, encoder[labels["label"][i]])

        if text_model: 
            data[mode]["text"] = data_text
        if audio_model: 
            data[mode]["audio"] = data_audio
        if vision_model: 
            data[mode]["vision"] = data_vision
        data[mode]["label"] = label

    if data_save_to:
        torch.save(data, data_save_to, pickle_protocol=4)

    return data

class dataset(Dataset):
    def __init__(self, data, mode):
        
        self.text = FloatTensor(data[mode]["text"])
        self.audio = FloatTensor(data[mode]["audio"])
        self.vision = FloatTensor(data[mode]["vision"])
        self.label = LongTensor(data[mode]["label"])
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):

        data = {
            'text': self.text[index],
            'audio': self.audio[index],
            'vision': self.vision[index],
        }
        label = self.label[index]
        
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
        self.best_score = None
        self.early_stop = False
        self.val_acc_max= 0
        os.makedirs(save_path, exist_ok=True)

    def __call__(self, val_acc, model):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping!")
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation accuracy improvement ({self.val_acc_max:.2%} --> {val_acc:.2%}).  Saving model ...')
        path = os.path.join(self.save_path, 'best.pth')
        torch.save(model, path)	
        self.val_acc_max = val_acc


def validation(model, test_loader):
    
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    model = model.to(device)
    
    running_loss = 0
    correct = 0
    total = 0

    all_pred = torch.tensor([]).to(device)
    all_true = torch.tensor([]).to(device)
    loss_function = CrossEntropyLoss()
    
    with torch.no_grad():
        
        model.eval()
        
        for data, label in tqdm(test_loader, desc="Valid", ncols=100, unit_scale=True, colour="red"):
            
            
            text = data["text"]
            audio = data["audio"]
            vision = data["vision"]
            
            outputs = model(text, audio, vision)
            
            loss = loss_function(outputs, label)
            running_loss += loss.item()
            
            predicted = torch.argmax(outputs, axis=1)
            correct += (predicted == label).sum().item()
            total += label.size(0)
            
            all_pred = torch.cat((all_pred, predicted), dim=0)
            all_true = torch.cat((all_true, label), dim=0)
            
            torch.cuda.empty_cache()
            
    loss = running_loss / len(test_loader)
    acc = correct / total

    return loss, acc, all_pred.cpu(), all_true.cpu()


def Train(model, train_loader, valid_loader, lr, eposhs, model_save_to):
    
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    model = model.to(device)
    
    optimizer = Adam(model.parameters(), lr=lr)
    loss_function = CrossEntropyLoss()
    early_stopping = EarlyStopping(model_save_to, patience=100)
    scheduler = ReduceLROnPlateau(optimizer, mode='max',factor=0.1 ,patience=50)
    
    history = {}
    history["train loss"] = []
    history["train acc"] = [] 
    history["valid loss"] = []
    history["valid acc"] = []

    for epoch in range(eposhs):
        
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for data, label in tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=100, unit_scale=True, colour="blue"):
           
            optimizer.zero_grad()
            
            text = data["text"]
            audio = data["audio"]
            vision = data["vision"]
            
            outputs = model(text, audio, vision)    
            
            loss = loss_function(outputs, label)
            
            running_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            predicted = torch.argmax(outputs, axis=1)
            correct += (predicted == label).sum().item()
                        
            total += label.size(0)
            
            torch.cuda.empty_cache()
            
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        val_loss, val_acc, _, _ = validation(model, valid_loader)
        scheduler.step(val_acc)
        print(f"Train Loss: {train_loss:.6f}\tTrain Acc: {train_acc:.2%}\tVal Loss: {val_loss:.6f}\tVal Acc: {val_acc:.2%}\tLR: {optimizer.state_dict()['param_groups'][0]['lr']}")
                
        history["train loss"].append(train_loss)
        history["train acc"].append(train_acc)
        history["valid loss"].append(val_loss)
        history["valid acc"].append(val_acc)
        
        early_stopping(val_acc, model)
        if early_stopping.early_stop: break
    
    return model, history

def visualization(history, pred, true):
    
    print("="*100)
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
    plt.title(f"Learning Curve Loss ({acc:.2%})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"learning_curves_loss")

    plt.figure()
    plt.plot(history["train acc"])
    plt.plot(history["valid acc"])
    plt.legend(["Train", "Valid"])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Learning Curve Accuracy ({acc:.2%})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"learning_curves_accuracy")

    cf_matrix = confusion_matrix(true, pred)
    df_cm = pd.DataFrame(cf_matrix, target_names, target_names)
    plt.figure(figsize = (9,6))
    heatmap(df_cm, annot=True, fmt="d")
    plt.xlabel("Prediction")
    plt.ylabel("Label")
    plt.title(f"Confusion Matrix ({acc:.2%})")
    plt.tight_layout()
    plt.show()
    plt.savefig(f"confusion_matrix")
