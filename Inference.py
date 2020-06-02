import torch
import numpy as np
import logging, yaml, os, sys, argparse, time
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
from scipy.io import wavfile

from Modules import Generator
from Datasets import InferenceDataset, Inference_Collater

with open('Hyper_Parameter.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

if not hp_Dict['Device'] is None:
    os.environ['CUDA_VISIBLE_DEVICES']= hp_Dict['Device']

if not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(0)

logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

class Inferencer:
    def __init__(self, checkpoint_Path):
        self.Model_Generate()        
        self.Load_Checkpoint(checkpoint_Path= checkpoint_Path)

    def Model_Generate(self):
        self.generator = Generator().to(device).eval()

    @torch.no_grad()
    def Inference_Step(self, files, mels, noises, result_Path):
        mels = mels.to(device)
        noises = noises.to(device)
        fakes = self.generator(noises, mels).cpu().numpy()

        os.makedirs(result_Path, exist_ok= True)

        for file, mel, fake in zip(files, mels.cpu().numpy(), fakes):
            new_Figure = plt.figure(figsize=(40, 10 * 2), dpi=100)
            plt.subplot(211)
            plt.imshow(mel, aspect='auto', origin='lower')
            plt.title('Mel spectrogram    File: {}'.format(file))
            plt.subplot(212)
            plt.plot(fake)
            plt.title('Inference wav    File: {}'.format(file))
            plt.tight_layout()
            plt.savefig(
                os.path.join(result_Path, file.replace('.npy', '.png')).replace("\\", "/")
                )
            plt.close(new_Figure)

            wavfile.write(
                filename= os.path.join(result_Path, file.replace('.npy', '.wav')).replace("\\", "/"),
                data= (fake * 32767.5).astype(np.int16),
                rate= hp_Dict['Sound']['Sample_Rate']
                )

    def Inference(
        self,
        mel_Path,
        result_Path= './results'
        ):
        logging.info('Mel-spectrogram path: {}'.format(mel_Path))
        logging.info('Result save path: {}'.format(result_Path))
        logging.info('Start inference.')

        os.makedirs(result_Path, exist_ok= True)

        dataLoader = torch.utils.data.DataLoader(
            dataset= InferenceDataset(
                mel_path= mel_Path
                ),
            shuffle= False,
            collate_fn= Inference_Collater(
                frame_Shift= hp_Dict['Sound']['Frame_Shift'],
                upsample_Pad= hp_Dict['WaveNet']['Upsample']['Pad'],
                max_Abs_Mel= hp_Dict['Sound']['Max_Abs_Mel']
                ),
            batch_size= hp_Dict['Train']['Batch_Size'],
            num_workers= hp_Dict['Train']['Num_Workers'],
            pin_memory= True
            )

        for files, mels, noises in tqdm(dataLoader, desc='[Inference]'):
            self.Inference_Step(files, mels, noises, result_Path)

    def Load_Checkpoint(self, checkpoint_Path):
        state_Dict = torch.load(
            checkpoint_Path,
            map_location= 'cpu'
            )
        self.generator.load_state_dict(state_Dict['Model']['Generator'])        

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-c', '--checkpoint', required= True)
    argParser.add_argument('-m', '--mel', required= True)
    argParser.add_argument('-r', '--result', default='./results')
    args = argParser.parse_args()
    
    new_Inferencer = Inferencer(checkpoint_Path= args.checkpoint)
    new_Inferencer.Inference(
        mel_Path= args.mel,
        result_Path= args.result
        )