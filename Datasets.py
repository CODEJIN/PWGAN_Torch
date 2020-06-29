import torch
import numpy as np
import pickle, os


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
        
        self.file_List = metadata_Dict['File_List']
        
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
            self.file_List = [line.strip().split('\t') for line in f.readlines()[1:]]

        self.pattern_Cache_Dict = {}
        

    def __getitem__(self, idx):
        if self.use_cache and idx in self.pattern_Cache_Dict.keys():
            return self.pattern_Cache_Dict[idx]

        audio, mel = Pattern_Generate(self.file_List[idx][1], top_db= 30)

        if self.use_cache:
            self.pattern_Cache_Dict[idx] = audio, mel, self.file_List[idx][0]

        return audio, mel, self.file_List[idx][0]

    def __len__(self):
        return len(self.file_List)

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        mel_path
        ):
        self.file_List = []
        for root, _, files in os.walk(mel_path):
            for file in files:
                if os.path.splitext(file)[1].upper() != '.NPY':
                    continue
                self.file_List.append(os.path.join(root, file).replace('\\', '/'))
        
    def __getitem__(self, idx):
        return os.path.basename(self.file_List[idx]), np.load(self.file_List[idx])

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
        self.mel_Length = wav_Length // frame_Shift

    def __call__(self, batch):        
        audios, mels = self.Stack(*zip(*batch))

        audios = torch.FloatTensor(audios)   # [Batch, Time]
        mels = torch.FloatTensor(mels).transpose(2, 1)   # [Batch, Time, Mel_dim] -> [Batch, Mel_dim, Time]
        noises = torch.randn(size= audios.size()) # [Batch, Time]

        return audios, mels, noises

    def Stack(self, audios, mels):
        audio_List = []
        mel_List = []
        for audio, mel in zip(audios, mels):
            mel_Pad = max(0, self.mel_Length + 2 * self.upsample_Pad - mel.shape[0])
            audio_Pad = max(0, self.wav_Length + 2 * self.upsample_Pad * self.frame_Shift - audio.shape[0])
            mel = np.pad(
                mel,
                [[int(np.floor(mel_Pad / 2)), int(np.ceil(mel_Pad / 2))], [0, 0]],
                mode= 'reflect'
                )            
            audio = np.pad(
                audio,
                [int(np.floor(audio_Pad / 2)), int(np.ceil(audio_Pad / 2))],
                mode= 'reflect'
                )

            mel_Offset = np.random.randint(self.upsample_Pad, max(mel.shape[0] - (self.mel_Length + self.upsample_Pad), self.upsample_Pad + 1))
            audio_Offset = mel_Offset * self.frame_Shift
            mel = mel[mel_Offset - self.upsample_Pad:mel_Offset + self.mel_Length + self.upsample_Pad]
            audio = audio[audio_Offset:audio_Offset + self.wav_Length]

            audio_List.append(audio)
            mel_List.append(mel)

        return np.stack(audio_List, axis= 0), np.stack(mel_List, axis= 0)

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
        max_Wav_Length = int(np.ceil(max([audio.shape[0] for audio, _, _ in batch]) / self.frame_Shift) * self.frame_Shift)
        wav_Length = np.minimum(self.wav_Length, max_Wav_Length)
        mel_Length = wav_Length // self.frame_Shift + self.upsample_Pad

        audios, mels, lengths, labels = [], [], [], []
        for index, (audio, mel, label) in enumerate(batch):
            length = audio.shape[0]
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
            lengths.append(length)
            labels.append(label)
            
        audios = torch.FloatTensor(np.stack(audios, axis= 0))   # [Batch, Time]
        mels = torch.FloatTensor(np.stack(mels, axis= 0)).transpose(2, 1)   # [Batch, Time, Mel_dim] -> [Batch, Mel_dim, Time]
        noises = torch.randn(size= audios.size()) # [Batch, Time]
        
        return audios, mels, noises, lengths, labels

class Inference_Collater:
    def __init__(
        self,
        frame_Shift,        
        upsample_Pad,
        max_Abs_Mel,
        ):
        self.frame_Shift = frame_Shift
        self.upsample_Pad = upsample_Pad
        self.max_Abs_Mel = max_Abs_Mel

    def __call__(self, batch):
        max_Mel_Length = max([mel.shape[0] for _, mel in batch])
        
        files = []
        mels = []
        for index, (file, mel) in enumerate(batch):            
            mel = np.pad(
                mel,
                pad_width=[[self.upsample_Pad, max_Mel_Length - mel.shape[0] + self.upsample_Pad], [0, 0]],
                constant_values= -self.max_Abs_Mel
                )
            files.append(file)
            mels.append(mel)
            
        mels = torch.FloatTensor(np.stack(mels, axis= 0)).transpose(2, 1)   # [Batch, Time, Mel_dim] -> [Batch, Mel_dim, Time]
        noises = torch.randn(size= (mels.size(0), max_Mel_Length * self.frame_Shift)) # [Batch, Time]
        
        return files, mels, noises


