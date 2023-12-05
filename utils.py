import os
import av
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from moviepy.editor import AudioFileClip, VideoFileClip
from transformers import AutoTokenizer, AutoProcessor, AutoModel

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
  
