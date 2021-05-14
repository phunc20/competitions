#def soundscapes_to_npy(is_test=False, n_processes=4):
#    pool = joblib.Parallel(n_processes)
#    mapping = joblib.delayed(every_5sec)
#    if is_test:
#        tasks = list(mapping(id_, save_to=testSoundScapes) for id_ in S_testSoundScapeIDs)
#        #tasks = list(mapping(id_,
#        #                     single_process=False,
#        #                     save_to=testSoundScapes)
#        #             for id_ in S_testSoundScapeIDs)
#    else:
#        tasks = list(mapping(id_, save_to=trainSoundScapes) for id_ in S_trainSoundScapeIDs)
#        #tasks = list(mapping(id_,
#        #                     single_process=False,
#        #                     save_to=trainSoundScapes)
#        #             for id_ in S_trainSoundScapeIDs)
#    pool(tqdm(tasks))




from pathlib import Path
from tqdm.notebook import tqdm
import librosa
import librosa.display
import numpy as np
import soundfile
import pandas as pd
import joblib

import random
import os

SR = 32_000
DURATION = 5
SEED = 42


class MelSpecComputer:
    def __init__(self, sr, n_mels, fmin, fmax, **kwargs):
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        kwargs["n_fft"] = kwargs.get("n_fft", self.sr//10)
        kwargs["hop_length"] = kwargs.get("hop_length", self.sr//(10*4))
        self.kwargs = kwargs

    def __call__(self, y):
        melspec = librosa.feature.melspectrogram(
            y,
            sr=self.sr,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            **self.kwargs,
        )

        melspec = librosa.power_to_db(melspec).astype(np.float32)
        return melspec

def standardize_uint8(X, eps=1e-6, mean=None, std=None):
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)
    
    min_, max_ = X.min(), X.max()
    if max_ - min_ > eps:
        #V = np.clip(X, min_, max_)
        #V = 255 * (V - min_) / (max_ - min_)
        V = 255 * (X - min_) / (max_ - min_)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)
    return V

def crop_or_pad(y, length, is_train=True, start=None):
    """
    crop or pad the signal y to #(samples) = `length`
      - repetition of itself
      - random truncating
    """
    if len(y) < length:
        #y = np.concatenate([y, np.zeros(length - len(y))])
        n_repeats = length // len(y)
        remainder = length % len(y)
        y = np.concatenate([y]*n_repeats + [y[:remainder]])
    elif len(y) > length:
        if not is_train:
            start = start or 0
        else:
            start = start or np.random.randint(len(y) - length)
        y = y[start:start + length]
    return y

def audio_to_mels(audio,
                  sr=SR,
                  n_mels=128,
                  fmin=0,
                  fmax=None):
    fmax = fmax or sr // 2
    mel_spec_computer = MelSpecComputer(sr=sr,
                                        n_mels=n_mels,
                                        fmin=fmin,
                                        fmax=fmax)
    mels = standardize_uint8(mel_spec_computer(audio))
    return mels





def every_5sec(id_,
               sr=SR,
               resample=True,
               res_type="kaiser_fast",
               single_process=True,
               save_to=Path("corbeille"),
               n_workers=2
                ):
    """
    - read the audio file of ID `id_`
    - cut the read audio into pieces of 5 seconds
    - convert each piece into `.npy` file and save
    """
    path_ogg = next((PATH_DATASET / "train_soundscapes").glob(f"{id_}*.ogg"))
    location = (path_ogg.name).split("_")[1]
    whole_audio, orig_sr = soundfile.read(path_ogg, dtype="float32")
    if resample and orig_sr != sr:
        whole_audio = librosa.resample(whole_audio, orig_sr, sr, res_type=res_type)
    n_samples = len(whole_audio)
    n_samples_5sec = sr * 5
    save_to.mkdir(exist_ok=True)

    def convert_and_save(i):
        audio_i = whole_audio[i:i + n_samples_5sec]
        mels_i = audio_to_mels(audio_i)
        path_i = save_to / f"{id_}_{location}_{((i + n_samples_5sec) // n_samples_5sec) * 5}.npy"
        np.save(str(path_i), mels_i)

    if single_process:
        for i in range(0, n_samples - n_samples % n_samples_5sec, n_samples_5sec):
            #audio_i = whole_audio[i:i + n_samples_5sec]
            ## No need the next check because in range() we have subtracted the remainder.
            ## That is, len(audio_i) is guaranteed to be n_samples_5sec for all i.
            ##if len(audio_i) < n_samples_5sec:
            ##    pass
            #mels_i = audio_to_mels(audio_i)
            #path_i = save_to / f"{id_}_{location}_{((i + n_samples_5sec) // n_samples_5sec) * 5}.npy"
            #np.save(str(path_i), mels_i)
            convert_and_save(i)
    else:
        pool = joblib.Parallel(n_workers)
        mapping = joblib.delayed(convert_and_save)
        tasks = (mapping(i) for i in range(0, n_samples - n_samples % n_samples_5sec, n_samples_5sec))
        pool(tasks)


def shortaudio_to_npy(L_ogg_paths,
                      n_processes=4,
                      save_to=Path("corbeille"),
    ):
    pool = joblib.Parallel(n_processes)
    mapping = joblib.delayed(into_5sec_npy)
    tasks = list(mapping(ogg_path, save_to=save_to) for ogg_path in L_ogg_paths)
    pool(tqdm(tasks))

def into_5sec_npy(ogg_path,
                  sr=SR,
                  resample=True,
                  res_type="kaiser_fast",
                  #single_process=True,
                  step_in_sec=5,
                  save_to=Path("corbeille"),
                  #n_workers=2,
                 ):
    """
    - read the audio file of ID `id_`
    - cut the read audio into pieces of 5 seconds
    - convert each piece into `.npy` file and save
    """
    ogg_filename = ogg_path.name
    sans_ext = ogg_filename.split(".")[0]
    whole_audio, orig_sr = soundfile.read(ogg_path, dtype="float32")
    if resample and orig_sr != sr:
        whole_audio = librosa.resample(whole_audio, orig_sr, sr, res_type=res_type)
    n_samples = len(whole_audio)
    n_samples_5sec = sr * 5
    n_samples_1step = sr * step_in_sec
    save_to.mkdir(exist_ok=True)

    #def convert_and_save(i):
    #    audio_i = whole_audio[i:i + n_samples_5sec]
    #    mels_i = audio_to_mels(audio_i)
    #    path_i = save_to / f"{id_}_{location}_{((i + n_samples_5sec) // n_samples_5sec) * 5}.npy"
    #    np.save(str(path_i), mels_i)

    for i in range(0, n_samples - n_samples_5sec, n_samples_1step):
        audio_i = whole_audio[i:i + n_samples_5sec]
        mels_i = audio_to_mels(audio_i)
        path_i = save_to / f"{sans_ext}_{((i + n_samples_5sec) // n_samples_5sec) * 5}.npy"
        np.save(path_i, mels_i)
        #np.save(str(path_i), mels_i)

