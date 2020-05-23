import torch
import numpy as np
import librosa, pickle, os


from Pattern_Generator import Pattern_Generate

class TrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pattern_path,
        metadata_file,
        wav_length,
        use_cache= False
        ):
        self.pattern_path = pattern_path        
        self.use_cache = use_cache

        with open(os.path.join(self.pattern_path, metadata_file).replace('\\', '/'), 'rb') as f:
            metadata_Dict = pickle.load(f)
        
        self.file_List = [
            file for file, length in metadata_Dict['Sig_Length_Dict'].items()
            if length > wav_length
            ]
        
        self.pattern_Cache_Dict = {}

    def __getitem__(self, idx):
        if idx in self.pattern_Cache_Dict.keys():
            return self.pattern_Cache_Dict[idx]['Signal'], self.pattern_Cache_Dict[idx]['Mel']

        with open(os.path.join(self.pattern_path, self.file_List[idx]).replace('\\', '/'), 'rb') as f:
            pattern_Dict = pickle.load(f)
        
        if self.use_cache:
            self.pattern_Cache_Dict[idx] = pattern_Dict

        return pattern_Dict['Signal'], pattern_Dict['Mel']

    def __len__(self):
        return len(self.file_List)

class DevDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pattern_path= 'Inference_Wav_for_Training.txt',
        use_cache= False
        ):
        self.use_cache = use_cache

        with open(pattern_path, 'r') as f:
            self.file_List = [line.strip() for line in f.readlines()]

        self.pattern_Cache_Dict = {}
        

    def __getitem__(self, idx):
        if self.use_cache and idx in self.pattern_Cache_Dict.keys():
            return self.pattern_Cache_Dict[idx]

        audio, mel = Pattern_Generate(self.file_List[idx])

        if self.use_cache:
            self.pattern_Cache_Dict[idx] = audio, mel

        return audio, mel

    def __len__(self):
        return len(self.file_List)

class Train_Collater:
    def __init__(
        self,
        wav_Length,
        frame_Shift,
        upsample_Pad,
        ):
        self.wav_Length = wav_Length
        self.frame_Shift = frame_Shift
        self.upsample_Pad = upsample_Pad
        self.mel_Length = wav_Length // frame_Shift + 2 * upsample_Pad

    def __call__(self, batch):        
        audios, mels = [], []
        for index, (audio, mel) in enumerate(batch):            
            max_Offset = mel.shape[0] - 2 - (self.mel_Length + 2 * self.upsample_Pad)
            if max_Offset <= 0:
                continue
                
            mel_Offset = np.random.randint(0, max_Offset)
            audio_Offset = (mel_Offset + self.upsample_Pad) * self.frame_Shift

            audio = audio[audio_Offset:audio_Offset + self.wav_Length]
            mel = mel[mel_Offset:mel_Offset + self.mel_Length]

            audios.append(audio)
            mels.append(mel)
                
        audios = torch.FloatTensor(np.stack(audios, axis= 0))   # [Batch, Time]
        mels = torch.FloatTensor(np.stack(mels, axis= 0)).transpose(2, 1)   # [Batch, Time, Mel_dim] -> [Batch, Mel_dim, Time]
        noises = torch.randn(size= audios.size()) # [Batch, Time]

        return audios, mels, noises

class Dev_Collater:
    def __init__(
        self,
        wav_Length,
        frame_Shift,
        upsample_Pad,
        max_Abs_Mel,
        ):
        self.wav_Length = wav_Length
        self.frame_Shift = frame_Shift
        self.upsample_Pad = upsample_Pad
        self.max_Abs_Mel = max_Abs_Mel

    def __call__(self, batch):
        max_Wav_Length = int(np.ceil(max([audio.shape[0] for audio, _ in batch]) / self.frame_Shift) * self.frame_Shift)
        wav_Length = np.minimum(self.wav_Length, max_Wav_Length)
        mel_Length = wav_Length // self.frame_Shift + self.upsample_Pad

        audios, mels = [], []
        for index, (audio, mel) in enumerate(batch):
            audio = audio[0:wav_Length]
            mel = mel[0:mel_Length]

            audio = np.pad(
                audio,
                pad_width=[0, wav_Length - audio.shape[0]],
                constant_values= 0
                )
            mel = np.pad(
                mel,
                pad_width=[[self.upsample_Pad, mel_Length - mel.shape[0]], [0, 0]],
                constant_values= -self.max_Abs_Mel
                )
            
            audios.append(audio)
            mels.append(mel)
            
        audios = torch.FloatTensor(np.stack(audios, axis= 0))   # [Batch, Time]
        mels = torch.FloatTensor(np.stack(mels, axis= 0)).transpose(2, 1)   # [Batch, Time, Mel_dim] -> [Batch, Mel_dim, Time]
        noises = torch.randn(size= audios.size()) # [Batch, Time]
        
        return audios, mels, noises


