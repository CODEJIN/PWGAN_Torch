import torch
import numpy as np
import logging, yaml, os, sys, argparse, time
from tqdm import tqdm
from collections import defaultdict
from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use('agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
from scipy.io import wavfile

from Modules import Generator, Discriminator, MultiResolutionSTFTLoss
from Datasets import TrainDataset, DevDataset, Train_Collater, Dev_Collater
from Radam import RAdam

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

class Trainer:
    def __init__(self, steps= 0):
        self.steps = steps
        self.epochs = 0

        self.Datset_Generate()
        self.Model_Generate()

        self.loss_Dict = {
            'Train': defaultdict(float),
            'Evaluation': defaultdict(float),
            }
        self.writer = SummaryWriter(hp_Dict['Log_Path'])

        if self.steps > 0:
            self.Load_Checkpoint()


    def Datset_Generate(self):
        train_Dataset = TrainDataset(
            pattern_path= hp_Dict['Train']['Pattern_Path'],
            metadata_file= hp_Dict['Train']['Metadata_File'],
            wav_length= hp_Dict['Train']['Wav_Length'],
            use_cache= hp_Dict['Train']['Use_Pattern_Cache'],
            )
        dev_Dataset = DevDataset(
            pattern_path= 'Inference_Wav_for_Training.txt',
            use_cache= hp_Dict['Train']['Use_Pattern_Cache'],
            )
        logging.info('The number of train files = {}.'.format(len(train_Dataset)))
        logging.info('The number of development files = {}.'.format(len(dev_Dataset)))

        train_Collater = Train_Collater(
            wav_Length= hp_Dict['Train']['Wav_Length'],
            frame_Shift= hp_Dict['Sound']['Frame_Shift'],
            upsample_Pad= hp_Dict['WaveNet']['Upsample']['Pad'],
            )
        dev_Collater = Dev_Collater(
            wav_Length= 15 * hp_Dict['Sound']['Sample_Rate'],
            frame_Shift= hp_Dict['Sound']['Frame_Shift'],
            upsample_Pad= hp_Dict['WaveNet']['Upsample']['Pad'],
            max_Abs_Mel= hp_Dict['Sound']['Max_Abs_Mel']
            )

        self.dataLoader_Dict = {}
        self.dataLoader_Dict['Train'] = torch.utils.data.DataLoader(
            dataset= train_Dataset,
            shuffle= True,
            collate_fn= train_Collater,
            batch_size= hp_Dict['Train']['Batch_Size'],
            num_workers= hp_Dict['Train']['Num_Workers'],
            pin_memory= True
            )
        self.dataLoader_Dict['Dev'] = torch.utils.data.DataLoader(
            dataset= dev_Dataset,
            shuffle= False,
            collate_fn= dev_Collater,
            batch_size= hp_Dict['Train']['Batch_Size'],
            num_workers= hp_Dict['Train']['Num_Workers'],
            pin_memory= True
            )

    def Model_Generate(self):
        self.model_Dict = {
            'Generator': Generator().to(device),
            'Discriminator': Discriminator().to(device)
            }
        self.criterion_Dict = {
            'STFT': MultiResolutionSTFTLoss(
                fft_sizes= hp_Dict['STFT_Loss_Resolution']['FFT_Sizes'],
                shift_lengths= hp_Dict['STFT_Loss_Resolution']['Shfit_Lengths'],
                win_lengths= hp_Dict['STFT_Loss_Resolution']['Win_Lengths'],
                ).to(device),
            'MSE': torch.nn.MSELoss().to(device)
            }
        self.optimizer_Dict = {
            'Generator': RAdam(
                params= self.model_Dict['Generator'].parameters(),
                lr= hp_Dict['Train']['Learning_Rate']['Generator']['Initial'],
                eps= hp_Dict['Train']['Learning_Rate']['Generator']['Epsilon'],
                ),
            'Discriminator': RAdam(
                params= self.model_Dict['Discriminator'].parameters(),
                lr= hp_Dict['Train']['Learning_Rate']['Discriminator']['Initial'],
                eps= hp_Dict['Train']['Learning_Rate']['Discriminator']['Epsilon'],
                )
            }
        self.scheduler_Dict = {
            'Generator': torch.optim.lr_scheduler.StepLR(
                optimizer= self.optimizer_Dict['Generator'],
                step_size= hp_Dict['Train']['Learning_Rate']['Generator']['Decay_Step'],
                gamma= hp_Dict['Train']['Learning_Rate']['Generator']['Decay_Rate'],
                ),
            'Discriminator': torch.optim.lr_scheduler.StepLR(
                optimizer= self.optimizer_Dict['Discriminator'],
                step_size= hp_Dict['Train']['Learning_Rate']['Discriminator']['Decay_Step'],
                gamma= hp_Dict['Train']['Learning_Rate']['Discriminator']['Decay_Rate'],
                )
            }
        logging.info(self.model_Dict['Generator'])
        logging.info(self.model_Dict['Discriminator'])


    def Train_Step(self, audios, mels, noises):
        loss_Dict = {}

        audios = audios.to(device)
        mels = mels.to(device)
        noises = noises.to(device)
        
        fake_Audios = self.model_Dict['Generator'](noises, mels)        
        
        loss_Dict['Spectral_Convergence'], loss_Dict['Magnitude'] = self.criterion_Dict['STFT'](fake_Audios, audios)
        loss_Dict['Generator'] = loss_Dict['Spectral_Convergence'] + loss_Dict['Magnitude']
        if self.steps > hp_Dict['Train']['Discriminator_Delay']:
            fake_Discriminations = self.model_Dict['Discriminator'](fake_Audios)
            loss_Dict['Adversarial'] = self.criterion_Dict['MSE'](
                fake_Discriminations,
                fake_Discriminations.new_ones(fake_Discriminations.size())
                )
            loss_Dict['Generator'] += hp_Dict['Train']['Adversarial_Weight'] * loss_Dict['Adversarial']
        
        
        self.optimizer_Dict['Generator'].zero_grad()
        loss_Dict['Generator'].backward()        
        torch.nn.utils.clip_grad_norm_(
            parameters= self.model_Dict['Generator'].parameters(),
            max_norm= hp_Dict['Train']['Generator_Gradient_Norm']
            )
        self.optimizer_Dict['Generator'].step()
        self.scheduler_Dict['Generator'].step()
                
        if self.steps > hp_Dict['Train']['Discriminator_Delay']:
            real_Discriminations = self.model_Dict['Discriminator'](audios)
            fake_Discriminations = self.model_Dict['Discriminator'](fake_Audios.detach())   #Why detached?

            loss_Dict['Real'] = self.criterion_Dict['MSE'](
                real_Discriminations,
                real_Discriminations.new_ones(real_Discriminations.size())
                )
            loss_Dict['Fake'] = self.criterion_Dict['MSE'](
                fake_Discriminations,
                fake_Discriminations.new_zeros(fake_Discriminations.size())
                )
            loss_Dict['Discriminator'] = loss_Dict['Real'] + loss_Dict['Fake']

            self.optimizer_Dict['Discriminator'].zero_grad()
            loss_Dict['Discriminator'].backward()
            torch.nn.utils.clip_grad_norm_(
                parameters= self.model_Dict['Discriminator'].parameters(),
                max_norm= hp_Dict['Train']['Discriminator_Gradient_Norm']
                )
            self.optimizer_Dict['Discriminator'].step()
            self.scheduler_Dict['Discriminator'].step()
        
        self.steps += 1
        self.tqdm.update(1)

        for tag, loss in loss_Dict.items():
            self.loss_Dict['Train'][tag] += loss

    def Train_Epoch(self):
        for step, (audios, mels, noises) in enumerate(self.dataLoader_Dict['Train'], 1):
            self.Train_Step(audios, mels, noises)
            
            if self.steps % hp_Dict['Train']['Checkpoint_Save_Interval'] == 0:
                self.Save_Checkpoint()

            if self.steps % hp_Dict['Train']['Logging_Interval'] == 0:
                self.loss_Dict['Train'] = {
                    tag: loss / hp_Dict['Train']['Logging_Interval']
                    for tag, loss in self.loss_Dict['Train'].items()
                    }
                self.Write_to_Tensorboard('Train', self.loss_Dict['Train'])
                self.loss_Dict['Train'] = defaultdict(float)

            if self.steps % hp_Dict['Train']['Evaluation_Interval'] == 0:
                self.Evaluation_Epoch()
            
            if self.steps >= hp_Dict['Train']['Max_Step']:
                return

        self.epochs += 1

    
    @torch.no_grad()
    def Evaluation_Step(self, audios, mels, noises):
        loss_Dict = {}

        audios = audios.to(device)
        mels = mels.to(device)
        noises = noises.to(device)

        fake_Audios = self.model_Dict['Generator'](noises, mels)
        loss_Dict['Spectral_Convergence'], loss_Dict['Magnitude'] = self.criterion_Dict['STFT'](fake_Audios, audios)
        loss_Dict['Generator'] = loss_Dict['Spectral_Convergence'] + loss_Dict['Magnitude']
        if self.steps > hp_Dict['Train']['Discriminator_Delay']:
            fake_Discriminations = self.model_Dict['Discriminator'](fake_Audios)
            loss_Dict['Adversarial'] = self.criterion_Dict['MSE'](
                fake_Discriminations,
                fake_Discriminations.new_ones(fake_Discriminations.size())
                )
            loss_Dict['Generator'] += hp_Dict['Train']['Discriminator_Delay'] * loss_Dict['Adversarial']
        
        if self.steps > hp_Dict['Train']['Discriminator_Delay']:
            real_Discriminations = self.model_Dict['Discriminator'](audios)
            fake_Discriminations = self.model_Dict['Discriminator'](fake_Audios.detach())   #Why detached?

            loss_Dict['Real'] = self.criterion_Dict['MSE'](
                real_Discriminations,
                real_Discriminations.new_ones(real_Discriminations.size())
                )
            loss_Dict['Fake'] = self.criterion_Dict['MSE'](
                fake_Discriminations,
                fake_Discriminations.new_zeros(fake_Discriminations.size())
                )
            loss_Dict['Discriminator'] = loss_Dict['Real'] + loss_Dict['Fake']

        for tag, loss in loss_Dict.items():
            self.loss_Dict['Evaluation'][tag] += loss

    @torch.no_grad()
    def Inference_Step(self, audios, mels, noises):
        mels = mels.to(device)
        noises = noises.to(device)
        fakes = self.model_Dict['Generator'](noises, mels).cpu().numpy()

        os.makedirs(os.path.join(hp_Dict['Inference_Path'], 'Step-{}'.format(self.steps)).replace("\\", "/"), exist_ok= True)

        for index, (real, fake) in enumerate(zip(audios, fakes)):            
            new_Figure = plt.figure(figsize=(80, 10 * 2), dpi=100)
            plt.subplot(211)
            plt.plot(real)
            plt.title('Original wav    Index: {}'.format(index))
            plt.subplot(212)            
            plt.plot(fake)
            plt.title('Fake wav    Index: {}'.format(index))
            plt.tight_layout()
            plt.savefig(
                os.path.join(hp_Dict['Inference_Path'], 'Step-{}'.format(self.steps), 'Step-{}.IDX_{}.PNG'.format(self.steps, index)).replace("\\", "/")
                )
            plt.close(new_Figure)

            wavfile.write(
                filename= os.path.join(hp_Dict['Inference_Path'], 'Step-{}'.format(self.steps), 'Step-{}.IDX_{}.WAV'.format(self.steps, index)).replace("\\", "/"),
                data= (fake * 32767.5).astype(np.int16),
                rate= hp_Dict['Sound']['Sample_Rate']
                )

    def Evaluation_Epoch(self):
        logging.info('(Steps: {}) Start evaluation.'.format(self.steps))

        for model in self.model_Dict.values():
            model.eval()

        for step, (audios, mels, noises) in tqdm(enumerate(self.dataLoader_Dict['Dev'], 1), desc='[Evaluation]'):
            self.Evaluation_Step(audios, mels, noises)
            self.Inference_Step(audios, mels, noises)

        self.loss_Dict['Evaluation'] = {
            tag: loss / step
            for tag, loss in self.loss_Dict['Evaluation'].items()
            }
        self.Write_to_Tensorboard('Evaluation', self.loss_Dict['Evaluation'])
        self.loss_Dict['Evaluation'] = defaultdict(float)
        
        for model in self.model_Dict.values():
            model.train()
        

    def Load_Checkpoint(self):
        state_Dict = torch.load(
            os.path.join(hp_Dict['Checkpoint_Path'], 'S_{}.pkl'.format(self.steps).replace('\\', '/')),
            map_location= 'cpu'
            )

        self.model_Dict['Generator'].load_state_dict(state_Dict['Model']['Generator'])
        self.model_Dict['Discriminator'].load_state_dict(state_Dict['Model']['Discriminator'])
        
        self.optimizer_Dict['Generator'].load_state_dict(state_Dict['Optimizer']['Generator'])
        self.optimizer_Dict['Discriminator'].load_state_dict(state_Dict['Optimizer']['Discriminator'])

        self.scheduler_Dict['Generator'].load_state_dict(state_Dict['Scheduler']['Generator'])
        self.scheduler_Dict['Discriminator'].load_state_dict(state_Dict['Scheduler']['Discriminator'])
        
        self.steps = state_Dict['Steps']
        self.epochs = state_Dict['Epochs']

        logging.info('Checkpoint loaded at {} steps.'.format(self.steps))

    def Save_Checkpoint(self):
        os.makedirs(hp_Dict['Checkpoint_Path'], exist_ok= True)

        state_Dict = {
            'Model': {
                'Generator': self.model_Dict['Generator'].state_dict(),
                'Discriminator': self.model_Dict['Discriminator'].state_dict(),
                },
            'Optimizer': {
                'Generator': self.optimizer_Dict['Generator'].state_dict(),
                'Discriminator': self.optimizer_Dict['Discriminator'].state_dict(),
                },
            'Scheduler': {
                'Generator': self.scheduler_Dict['Generator'].state_dict(),
                'Discriminator': self.scheduler_Dict['Discriminator'].state_dict(),
                },
            'Steps': self.steps,
            'Epochs': self.epochs,
            }

        torch.save(
            state_Dict,
            os.path.join(hp_Dict['Checkpoint_Path'], 'S_{}.pkl'.format(self.steps).replace('\\', '/'))
            )

        logging.info('Checkpoint saved at {} steps.'.format(self.steps))
       

    def Train(self):
        self.tqdm = tqdm(
            initial= self.steps,
            total= hp_Dict['Train']['Max_Step'],
            desc='[Training]'
            )

        while self.steps < hp_Dict['Train']['Max_Step']:
            try:
                self.Train_Epoch()
            except KeyboardInterrupt:
                self.Save_Checkpoint()
                exit(1)
            
        self.tqdm.close()
        logging.info('Finished training.')

    def Write_to_Tensorboard(self, category, loss_Dict):
        for tag, loss in loss_Dict.items():
            self.writer.add_scalar('{}/{}'.format(category, tag), loss, self.steps)

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', '--steps', default= 0, type= int)
    args = argParser.parse_args()
    
    new_Trainer = Trainer(steps= args.steps)    
    new_Trainer.Train()