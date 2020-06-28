import numpy as np
import yaml, os, time, pickle, librosa, re, argparse
from concurrent.futures import ThreadPoolExecutor as PE
from collections import deque
from threading import Thread
from random import shuffle

from Audio import Audio_Prep, Mel_Generate

with open('Hyper_Parameter.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

using_Extension = [x.upper() for x in ['.wav', '.m4a', '.flac']]

def Pattern_Generate(path, top_db= 60):
    sig = Audio_Prep(path, hp_Dict['Sound']['Sample_Rate'], top_db)
    mel = Mel_Generate(
        audio= sig,
        sample_rate= hp_Dict['Sound']['Sample_Rate'],
        num_frequency= hp_Dict['Sound']['Spectrogram_Dim'],
        num_mel= hp_Dict['Sound']['Mel_Dim'],
        window_length= hp_Dict['Sound']['Frame_Length'],
        hop_length= hp_Dict['Sound']['Frame_Shift'],
        mel_fmin= hp_Dict['Sound']['Mel_F_Min'],
        mel_fmax= hp_Dict['Sound']['Mel_F_Max'],
        max_abs_value= hp_Dict['Sound']['Max_Abs_Mel']
        )

    return sig, mel

def Pattern_File_Generate(path, speaker_ID, dataset, file_Prefix='', display_Prefix = '', top_db= 60):
    sig, mel = Pattern_Generate(path, top_db)
    
    new_Pattern_Dict = {
        'Signal': sig.astype(np.float32),
        'Mel': mel.astype(np.float32),
        'Speaker_ID': speaker_ID,
        'Dataset': dataset,
        }

    pickle_File_Name = '{}.{}{}.PICKLE'.format(dataset, file_Prefix, os.path.splitext(os.path.basename(path))[0]).upper()

    with open(os.path.join(hp_Dict['Train']['Pattern_Path'], pickle_File_Name).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_Pattern_Dict, f, protocol=2)
            
    print('[{}]'.format(display_Prefix), '{}'.format(path), '->', '{}'.format(pickle_File_Name))

def Speaker_Index_Dict_Generate(lj, bc2013, fv, vctk):
    speaker_Index_Dict = {}
    current_Index = 0
    if lj:
        speaker_Index_Dict['LJ'] = current_Index
        current_Index += 1
    if bc2013:
        speaker_Index_Dict['BC2013'] = current_Index
        current_Index += 1
    if fv:
        for index, speaker in enumerate(['FV.AWB', 'FV.BDL', 'FV.CLB', 'FV.JMK', 'FV.KSP', 'FV.RMS', 'FV.SLT']):
            speaker_Index_Dict[speaker] = current_Index + index

    if vctk:
        for index, speaker in enumerate([
            'VCTK.P233', 'VCTK.P234', 'VCTK.P236', 'VCTK.P237', 'VCTK.P238', 'VCTK.P239', 'VCTK.P240', 'VCTK.P241',
            'VCTK.P243', 'VCTK.P244', 'VCTK.P245', 'VCTK.P246', 'VCTK.P247', 'VCTK.P248', 'VCTK.P249', 'VCTK.P250',
            'VCTK.P251', 'VCTK.P252', 'VCTK.P253', 'VCTK.P254', 'VCTK.P255', 'VCTK.P256', 'VCTK.P257', 'VCTK.P258',
            'VCTK.P259', 'VCTK.P260', 'VCTK.P261', 'VCTK.P262', 'VCTK.P263', 'VCTK.P264', 'VCTK.P265', 'VCTK.P266',
            'VCTK.P267', 'VCTK.P268', 'VCTK.P269', 'VCTK.P270', 'VCTK.P271', 'VCTK.P272', 'VCTK.P273', 'VCTK.P274',
            'VCTK.P275', 'VCTK.P276', 'VCTK.P277', 'VCTK.P278', 'VCTK.P279', 'VCTK.P280', 'VCTK.P281', 'VCTK.P282',
            'VCTK.P283', 'VCTK.P284', 'VCTK.P285', 'VCTK.P286', 'VCTK.P287', 'VCTK.P288', 'VCTK.P292', 'VCTK.P293',
            'VCTK.P294', 'VCTK.P295', 'VCTK.P297', 'VCTK.P298', 'VCTK.P299', 'VCTK.P300', 'VCTK.P301', 'VCTK.P302',
            'VCTK.P303', 'VCTK.P304', 'VCTK.P305', 'VCTK.P306', 'VCTK.P307', 'VCTK.P308', 'VCTK.P310', 'VCTK.P311',
            'VCTK.P312', 'VCTK.P313', 'VCTK.P314', 'VCTK.P315', 'VCTK.P316', 'VCTK.P317', 'VCTK.P318', 'VCTK.P323',
            'VCTK.P326', 'VCTK.P329', 'VCTK.P330', 'VCTK.P333', 'VCTK.P334', 'VCTK.P335', 'VCTK.P336', 'VCTK.P339',
            'VCTK.P340', 'VCTK.P341', 'VCTK.P343', 'VCTK.P345', 'VCTK.P347', 'VCTK.P351', 'VCTK.P360', 'VCTK.P361',
            'VCTK.P362', 'VCTK.P363', 'VCTK.P364', 'VCTK.P374', 'VCTK.P376', 'VCTK.P225', 'VCTK.P226', 'VCTK.P227',
            'VCTK.P228', 'VCTK.P229', 'VCTK.P230', 'VCTK.P231', 'VCTK.P232',
            ]):
            speaker_Index_Dict[speaker] = current_Index + index

    return speaker_Index_Dict


def LJ_Info_Load(lj_Path):
    lj_File_Path_List = []

    for root, _, file_Name_List in os.walk(lj_Path):
        for file_Name in file_Name_List:
            wav_File_Path = os.path.join(root, file_Name).replace('\\', '/')
            if not os.path.splitext(wav_File_Path)[1].upper() in using_Extension:
                continue
            lj_File_Path_List.append(wav_File_Path)
            
    print('LJ info generated: {}'.format(len(lj_File_Path_List)))
    return lj_File_Path_List

def BC2013_Info_Load(bc2013_Path):
    text_Path_List = []
    for root, _, files in os.walk(bc2013_Path):
        for filename in files:
            if os.path.splitext(filename)[1].upper() != '.txt'.upper():
                continue
            text_Path_List.append(os.path.join(root, filename).replace('\\', '/'))

    bc2013_File_Path_List = []

    for text_Path in text_Path_List:
        wav_Path = text_Path.replace('txt', 'wav')
        if not os.path.exists(wav_Path):
            continue
        bc2013_File_Path_List.append(wav_Path)

    print('BC2013 info generated: {}'.format(len(bc2013_File_Path_List)))
    return bc2013_File_Path_List

def FV_Info_Load(fv_Path):
    text_Path_List = []
    for root, _, file_Name_List in os.walk(fv_Path):
        for file in file_Name_List:
            if os.path.splitext(file)[1] == '.data':
                text_Path_List.append(os.path.join(root, file).replace('\\', '/'))

    fv_File_Path_List = []
    fv_Speaker_Dict = {}
    for text_Path in text_Path_List:        
        speaker = text_Path.split('/')[-3].split('_')[2].upper()
        with open(text_Path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            file_Path, _, _ = line.strip().split('"')

            file_Path = file_Path.strip().split(' ')[1]
            wav_File_Path = os.path.join(
                os.path.split(text_Path)[0].replace('etc', 'wav'),
                '{}.wav'.format(file_Path)
                ).replace('\\', '/')

            fv_File_Path_List.append(wav_File_Path)
            fv_Speaker_Dict[wav_File_Path] = speaker

    print('FV info generated: {}'.format(len(fv_File_Path_List)))
    return fv_File_Path_List, fv_Speaker_Dict

def VCTK_Info_Load(vctk_Path):
    vctk_Wav_Path = os.path.join(vctk_Path, 'wav48').replace('\\', '/')
    try:
        with open(os.path.join(vctk_Path, 'VCTK.NonOutlier.txt').replace('\\', '/'), 'r') as f:
            vctk_Non_Outlier_List = [x.strip() for x in f.readlines()]
    except:
        vctk_Non_Outlier_List = None

    vctk_File_Path_List = []
    for root, _, files in os.walk(vctk_Wav_Path):
        for file in files:
            if not vctk_Non_Outlier_List is None and not file in vctk_Non_Outlier_List:
                continue
            wav_File_Path = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(wav_File_Path)[1].upper() in using_Extension:
                continue

            vctk_File_Path_List.append(wav_File_Path)

    vctk_Speaker_Dict = {
        path: path.split('/')[-2].upper()
        for path in vctk_File_Path_List
        }

    print('VCTK info generated: {}'.format(len(vctk_File_Path_List)))
    return vctk_File_Path_List, vctk_Speaker_Dict



def Metadata_Generate():
    new_Metadata_Dict = {
        'Spectrogram_Dim': hp_Dict['Sound']['Spectrogram_Dim'],
        'Mel_Dim': hp_Dict['Sound']['Mel_Dim'],
        'Frame_Shift': hp_Dict['Sound']['Frame_Shift'],
        'Frame_Length': hp_Dict['Sound']['Frame_Length'],
        'Sample_Rate': hp_Dict['Sound']['Sample_Rate'],
        'Max_Abs_Mel': hp_Dict['Sound']['Max_Abs_Mel'],
        'File_List': [],
        'Sig_Length_Dict': {},
        'Mel_Length_Dict': {},
        'Speaker_ID_Dict': {},
        'Dataset_Dict': {},
        }

    for root, _, files in os.walk(hp_Dict['Train']['Pattern_Path']):
        for file in files:
            with open(os.path.join(root, file).replace("\\", "/"), "rb") as f:
                pattern_Dict = pickle.load(f)
                try:
                    new_Metadata_Dict['Sig_Length_Dict'][file] = pattern_Dict['Signal'].shape[0]
                    new_Metadata_Dict['Mel_Length_Dict'][file] = pattern_Dict['Mel'].shape[0]
                    new_Metadata_Dict['Speaker_ID_Dict'][file] = pattern_Dict['Speaker_ID']
                    new_Metadata_Dict['Dataset_Dict'][file] = pattern_Dict['Dataset']
                    new_Metadata_Dict['File_List'].append(file)
                except:
                    print('File \'{}\' is not correct pattern file. This file is ignored.'.format(file))

    with open(os.path.join(hp_Dict['Train']['Pattern_Path'], hp_Dict['Train']['Metadata_File'].upper()).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_Metadata_Dict, f, protocol=2)

    print('Metadata generate done.')

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-lj", "--lj_path", required=False)
    argParser.add_argument("-bc2013", "--bc2013_path", required=False)
    argParser.add_argument("-fv", "--fv_path", required=False)
    argParser.add_argument("-vctk", "--vctk_path", required=False)
    argParser.add_argument("-mc", "--max_count", required=False)
    argParser.add_argument("-mw", "--max_worker", required=False)
    argParser.set_defaults(max_worker = 10)
    argument_Dict = vars(argParser.parse_args())
    
    if not argument_Dict['max_count'] is None:
        argument_Dict['max_count'] = int(argument_Dict['max_count'])

    total_Pattern_Count = 0

    if not argument_Dict['lj_path'] is None:
        lj_File_Path_List = LJ_Info_Load(lj_Path= argument_Dict['lj_path'])
        total_Pattern_Count += len(lj_File_Path_List)
    if not argument_Dict['bc2013_path'] is None:
        bc2013_File_Path_List = BC2013_Info_Load(bc2013_Path= argument_Dict['bc2013_path'])
        total_Pattern_Count += len(bc2013_File_Path_List)
    if not argument_Dict['fv_path'] is None:
        fv_File_Path_List, fv_Speaker_Dict = FV_Info_Load(fv_Path= argument_Dict['fv_path'])
        total_Pattern_Count += len(fv_File_Path_List)
    if not argument_Dict['vctk_path'] is None:
        vctk_File_Path_List, vctk_Speaker_Dict = VCTK_Info_Load(vctk_Path= argument_Dict['vctk_path'])
        total_Pattern_Count += len(vctk_File_Path_List)

    if total_Pattern_Count == 0:
        raise ValueError('Total pattern count is zero.')
    
    speaker_Index_Dict = Speaker_Index_Dict_Generate(
        lj= bool(argument_Dict['lj_path']),
        bc2013= bool(argument_Dict['bc2013_path']),
        fv= bool(argument_Dict['fv_path']),
        vctk= bool(argument_Dict['vctk_path']),
        )

    os.makedirs(hp_Dict['Train']['Pattern_Path'], exist_ok= True)
    total_Generated_Pattern_Count = 0
    with PE(max_workers = int(argument_Dict['max_worker'])) as pe:
        if not argument_Dict['lj_path'] is None:            
            for index, file_Path in enumerate(lj_File_Path_List):
                pe.submit(
                    Pattern_File_Generate,
                    file_Path,
                    speaker_Index_Dict['LJ'],
                    'LJ',
                    '',
                    'LJ {:05d}/{:05d}    Total {:05d}/{:05d}'.format(
                        index + 1,
                        len(lj_File_Path_List),
                        total_Generated_Pattern_Count,
                        total_Pattern_Count
                        ),
                    60
                    )
                total_Generated_Pattern_Count += 1

        if not argument_Dict['bc2013_path'] is None:
            for index, file_Path in enumerate(bc2013_File_Path_List):
                pe.submit(
                    Pattern_File_Generate,
                    file_Path,
                    speaker_Index_Dict['BC2013'],
                    'BC2013',
                    '{}.'.format(file_Path.split('/')[-2]),
                    'BC2013 {:05d}/{:05d}    Total {:05d}/{:05d}'.format(
                        index + 1,
                        len(bc2013_File_Path_List),
                        total_Generated_Pattern_Count,
                        total_Pattern_Count
                        ),
                    60
                    )
                total_Generated_Pattern_Count += 1

        if not argument_Dict['fv_path'] is None:
            for index, file_Path in enumerate(fv_File_Path_List):
                pe.submit(
                    Pattern_File_Generate,
                    file_Path,
                    speaker_Index_Dict['FV.{}'.format(fv_Speaker_Dict[file_Path])],
                    'FV',                    
                    '{}.'.format(fv_Speaker_Dict[file_Path]),
                    'FV {:05d}/{:05d}    Total {:05d}/{:05d}'.format(
                        index + 1,
                        len(fv_File_Path_List),
                        total_Generated_Pattern_Count,
                        total_Pattern_Count
                        ),
                    60
                    )
                total_Generated_Pattern_Count += 1

        if not argument_Dict['vctk_path'] is None:
            for index, file_Path in enumerate(vctk_File_Path_List):
                pe.submit(
                    Pattern_File_Generate,
                    file_Path,
                    speaker_Index_Dict['VCTK.{}'.format(vctk_Speaker_Dict[file_Path])],
                    'VCTK',                    
                    '{}.'.format(vctk_Speaker_Dict[file_Path]),
                    'VCTK {:05d}/{:05d}    Total {:05d}/{:05d}'.format(
                        index + 1,
                        len(vctk_File_Path_List),
                        total_Generated_Pattern_Count,
                        total_Pattern_Count
                        ),
                    15
                    )
                total_Generated_Pattern_Count += 1

    Metadata_Generate()