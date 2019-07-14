import os
import re
import audio_utilities
from python_speech_features import *
import librosa
import scipy.io.wavfile as wav
import numpy as np
from pylab import *
import random

#dir='./WCE-SLT-LIG/WAV_TRANSCRIPTION/wav_transcription_tst'
fr_dir='./fr_test'
en_dir='./en_test'
wav_n_fft=2048

def get_mfcc(data,fs):
    wav_feature=mfcc(data,fs)
    d_mfcc_feat = delta(wav_feature, 1)
    d_mfcc_feat2 = delta(wav_feature, 2)
    feature = np.hstack((wav_feature, d_mfcc_feat, d_mfcc_feat2))
    return feature

def get_full_abs_spec(signal,fft_size):
    hopsamp = fft_size // 8
    stft_full = audio_utilities.stft_for_reconstruction(signal,fft_size, hopsamp)
    stft_mag = abs(stft_full)**2.0
    scale = 1.0 / np.amax(stft_mag)
    stft_mag *= scale
    return stft_mag

def get_mel(data,fs,n_fft):
    melW=librosa.filters.mel(fs,n_fft=n_fft,n_mels=80,fmin=70,fmax=8000)
    melW /=np.max(melW,axis=-1)[:,None]
    melX=np.dot(data,melW.T)
    logmelX=np.log(melX)
    return melX,logmelX


def get_wav_files(wav_path,add_path):
    wav_files=[]
    temp_file=os.listdir(wav_path)
    temp_file.sort()
    for filename in temp_file:
        if filename.endswith('.wav') or filename.endswith('.WAV'):
            #filename_path=os.path.join('./WCE-SLT-LIG/WAV_TRANSCRIPTION/wav_transcription_tst',filename)
            filename_path=os.path.join(add_path,filename)
            wav_files.append(filename_path)
    return wav_files
def get_tran_texts(wav_files,tran_path):
    tran_texts=[]
    for wav_file in wav_files:
        (wav_path,wav_filename)=os.path.split(wav_file)
        tran_file=os.path.join(tran_path,wav_filename.split('.')[0]+'.transcription')
        if os.path.exists(tran_file) is False:
            return None

        fd=open(tran_file,'r')
        text=fd.readline()
        text=re.findall(".*<s>(.*)</s>.*",text)[0]
        text=text.strip()
        tran_texts.append(text.split('\n')[0])
        fd.close()
    return tran_texts

def get_wav_files_and_tran_texts(wav_path,tran_path):
    wave_files=get_wav_files(wav_path,fr_dir)
    tran_texts=get_tran_texts(wave_files,tran_path)
    return wave_files,tran_texts

def wave_file_to_mel(wave_files):
    log_mel_file=[]
    n=0
    max_l=0
    for wave_file in wave_files:
        fs,signal=wav.read(wave_file)
        stft_mag=get_full_abs_spec(signal,wav_n_fft)
        mel_spec,log_mel_spec=get_mel(stft_mag,fs,wav_n_fft)
        print(n+1,':',mel_spec.shape)
        '''
        if (n%100)==1:
            figure(n+1)
            imshow((mel_spec)**0.125, origin='lower', cmap=cm.hot, aspect='auto',interpolation='nearest')
            colorbar()
            title('Mel scale spectrogram')
            xlabel('time index')
            ylabel('mel frequency bin index')
            savefig('mel_scale_spectrogram'+str(n+1)+'.png', dpi=150)
        '''
        n=n+1
        log_mel_file.append(log_mel_spec)
    for i in range(len(log_mel_file)):
        print(log_mel_file[i].shape[0])
        if log_mel_file[i].shape[0]>=max_l:
            max_l=log_mel_file[i].shape[0]
        print("max length:",max_l)

    for i in range(len(log_mel_file)):
        log_mel_file[i]=np.pad(log_mel_file[i],((0,max_l-log_mel_file[i].shape[0]),(0,0)),'constant',constant_values=(0,0))
    return log_mel_file

class mel_dataset:
    def __init__(self):
        self.fr_wav_files,self.tran_texts=get_wav_files_and_tran_texts(fr_dir,fr_dir)
        self.en_wav_files=get_wav_files(en_dir,en_dir)
        print(len(self.fr_wav_files), len(self.tran_texts))
        self.fr_log_mel_spectrogram=wave_file_to_mel(self.fr_wav_files)
        self.en_log_mel_spectrogram=wave_file_to_mel(self.en_wav_files)
        self.batch_number=0

    def get_next_batch(self,batch_size=2):
        if(self.batch_number+batch_size<=len(self.en_log_mel_spectrogram)):
            x_batch=self.fr_log_mel_spectrogram[self.batch_number:self.batch_number+batch_size]
            y_batch=self.en_log_mel_spectrogram[self.batch_number:self.batch_number+batch_size]
            self.batch_number+=batch_size
        else:
            self.shuffle()
            self.batch_number=0
            x_batch=self.fr_log_mel_spectrogram[self.batch_number:self.batch_number+batch_size]
            y_batch=self.en_log_mel_spectrogram[self.batch_number:self.batch_number+batch_size]
        return x_batch,y_batch

    def shuffle(self):
        randnum = random.randint(0,100)
        random.seed(randnum)
        random.shuffle(self.fr_log_mel_spectrogram)
        random.seed(randnum)
        random.shuffle(self.en_log_mel_spectrogram)
        

