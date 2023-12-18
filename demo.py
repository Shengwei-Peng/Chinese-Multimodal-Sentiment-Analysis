import av
import torch
import whisper
import librosa
import numpy as np
import gradio as gr
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoProcessor, AutoModel, AutoFeatureExtractor

import warnings
warnings.filterwarnings('ignore')


class MSA():
    def __init__(self, path:str):

        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

        self.whisper = whisper.load_model("medium").to(self.device)
        self.options = whisper.DecodingOptions(language = 'zh')

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.text_model = AutoModel.from_pretrained("bert-base-chinese").to(self.device)

        self.audio_processor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.audio_model = AutoModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(self.device)

        self.vision_processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
        self.vision_model = AutoModel.from_pretrained("microsoft/xclip-base-patch32").to(self.device)

        self.model = torch.load(path)
        print("System startup completed!!!")

    def sample_frame_indices(self, clip_len, frame_sample_rate, seg_len):
    
        converted_len = int(clip_len * frame_sample_rate)
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        
        return indices


    def read_video_pyav(self, container, indices):

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
        data = list(data)

        return data

    def predict(self, inp):

        vision = av.open(inp)
        audio = whisper.load_audio(inp)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.device)
        text = whisper.decode(self.whisper, mel, self.options).text

        text = self.tokenizer(text, max_length=49, padding='max_length', return_tensors="pt").to(self.device)
        audio = self.audio_processor(audio, sampling_rate=16000, return_tensors="pt").to(self.device)
        indices = self.sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=vision.streams.video[0].frames)
        vision = self.read_video_pyav(vision, indices)
        vision = self.vision_processor(videos=vision, return_tensors="pt").to(self.device)
        label = {0:"Negative", 1:"Neutral", 2:"Positive"}

        with torch.no_grad():
            text = self.text_model(**text).last_hidden_state
            audio = self.audio_model(**audio).last_hidden_state
            vision = self.vision_model.get_video_features(**vision)[None, :]
            output = self.model(text, audio, vision)
        pred = torch.argmax(output, axis=1).cpu().numpy()[0]

        return inp, label[pred]

if __name__ == "__main__":

    
    model = MSA("fusion_model.pth")
    inputs = gr.Video(include_audio = True)
    gr.Interface(fn=model.predict, inputs=inputs, outputs=[gr.Video(),"text"]).launch(share=True)
