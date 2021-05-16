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
from collections import namedtuple


SR = 32_000
DURATION = 5
SEED = 42
PATH_DATASET = Path("../input/birdclef-2021")

L_birds = [path.name for path
           in (PATH_DATASET / "train_short_audio").iterdir()]
L_birds = sorted(L_birds)
D_label_index = {label: i for i, label in enumerate(L_birds)}
D_index_label = {v: k for k, v in D_label_index.items()}

Coordinate = namedtuple("Coordinate", ["longitude", "latitude"])

D_location_coordinate = dict()
for p in (PATH_DATASET / "test_soundscapes").glob("*_recording_location.txt"):
    location = p.stem.split("_")[0]
    with open(p) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Latitude:"):
                latitude = float(line.split(" ")[1])
            if line.startswith("Longitude:"):
                longitude = float(line.split(" ")[1])
    D_location_coordinate[location] = Coordinate(longitude=longitude, latitude=latitude)

def birdLabel_to_nBirds(label):
    if label == "nocall":
        return 0
    return len(label.split())

def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True

def duration(path_ogg):
    audio, orig_sr = soundfile.read(path_ogg, dtype="float32")
    audio = librosa.resample(audio, orig_sr, SR, res_type="kaiser_fast")
    return len(audio) / SR


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

#def every_5sec(id_,
#               sr=SR,
#               resample=True,
#               res_type="kaiser_fast",
#               single_process=True,
#               save_to=Path("corbeille"),
#               n_workers=2
#                ):
#    """
#    - read the audio file of ID `id_`
#    - cut the read audio into pieces of 5 seconds
#    - convert each piece into `.npy` file and save
#    """
#    path_ogg = next((PATH_DATASET / "train_soundscapes").glob(f"{id_}*.ogg"))
#    location = (path_ogg.name).split("_")[1]
#    whole_audio, orig_sr = soundfile.read(path_ogg, dtype="float32")
#    if resample and orig_sr != sr:
#        whole_audio = librosa.resample(whole_audio, orig_sr, sr, res_type=res_type)
#    n_samples = len(whole_audio)
#    n_samples_5sec = sr * 5
#    save_to.mkdir(exist_ok=True)
#
#    def convert_and_save(i):
#        audio_i = whole_audio[i:i + n_samples_5sec]
#        mels_i = audio_to_mels(audio_i)
#        path_i = save_to / f"{id_}_{location}_{((i + n_samples_5sec) // n_samples_5sec) * 5}.npy"
#        np.save(str(path_i), mels_i)
#
#    if single_process:
#        for i in range(0, n_samples - n_samples % n_samples_5sec, n_samples_5sec):
#            #audio_i = whole_audio[i:i + n_samples_5sec]
#            ## No need the next check because in range() we have subtracted the remainder.
#            ## That is, len(audio_i) is guaranteed to be n_samples_5sec for all i.
#            ##if len(audio_i) < n_samples_5sec:
#            ##    pass
#            #mels_i = audio_to_mels(audio_i)
#            #path_i = save_to / f"{id_}_{location}_{((i + n_samples_5sec) // n_samples_5sec) * 5}.npy"
#            #np.save(str(path_i), mels_i)
#            convert_and_save(i)
#    else:
#        pool = joblib.Parallel(n_workers)
#        mapping = joblib.delayed(convert_and_save)
#        tasks = (mapping(i) for i in range(0, n_samples - n_samples % n_samples_5sec, n_samples_5sec))
#        pool(tasks)
#
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


def audios_to_npy(L_ogg_paths,
                  n_processes=4,
                  sr=SR,
                  resample=True,
                  res_type="kaiser_fast",
                  is_soundscape=False,
                  step_in_sec=5,
                  save_to=Path("corbeille"),
    ):
    pool = joblib.Parallel(n_processes)
    mapping = joblib.delayed(save_into_5sec_npy)
    tasks = list(mapping(ogg_path,
                         sr=sr,
                         resample=resample,
                         res_type=res_type,
                         is_soundscape=is_soundscape,
                         step_in_sec=step_in_sec,
                         save_to=save_to,)
                 for ogg_path in L_ogg_paths)
    pool(tqdm(tasks))

def save_into_5sec_npy(ogg_path,
                       sr=SR,
                       resample=True,
                       res_type="kaiser_fast",
                       is_soundscape=False,
                       step_in_sec=5,
                       save_to=Path("corbeille"),
    ):
    """
    - read the audio file of ID `id_`
    - cut the read audio into pieces of 5 seconds
    - convert each piece into `.npy` file and save
    """
    ogg_filename = ogg_path.name
    if is_soundscape:
        # e.g.
        # 2782_SSW_20170701.ogg
        sans_ext = (ogg_filename.split(".")[0])[:-9]
        #sans_ext = "_".join(ogg_filename.split("_")[:2])
    else:
        # e.g.
        # XC247837.ogg
        sans_ext = ogg_filename.split(".")[0]
    whole_audio, orig_sr = soundfile.read(ogg_path, dtype="float32")
    if resample and orig_sr != sr:
        whole_audio = librosa.resample(whole_audio, orig_sr, sr, res_type=res_type)
    n_samples = len(whole_audio)
    n_samples_5sec = sr * 5
    n_samples_1step = sr * step_in_sec
    save_to.mkdir(exist_ok=True)

    #for i in range(0, n_samples, n_samples_1step):
    for i in range(0, n_samples - n_samples_5sec + 1, n_samples_1step):
        audio_i = whole_audio[i:i + n_samples_5sec]
        mels_i = audio_to_mels(audio_i)
        path_i = save_to / f"{sans_ext}_{((i + n_samples_5sec) // n_samples_5sec) * 5}.npy"
        np.save(path_i, mels_i)

def birds_to_ndarray(series):
    I = np.eye(len(D_label_index))
    ndarray = np.zeros((len(series), len(D_label_index)))
    for i, string in enumerate(series.values):
        if string == "nocall":
            continue
        else:
            L_indices = [D_label_index[label] for label in string.split(" ")]
            row_i = np.sum(I[L_indices], axis=0)
            ndarray[i] = row_i
    return ndarray

def cyclicize_number(number, max_, min_):
    """
    args
        number, int
            \in {min_, min_ + 1, ..., max_}
            e.g. hour => min_ = 0, max_ = 24
                 longitude => min_ = -180, max_ = 180
        max_, int
        min_, int
    return
        (x, y), tuple of float
    """
    period = max_ - min_
    theta = 2 * np.pi * (number / period)
    #theta = 2 * np.pi * ((number - min_) / period)
    x = np.cos(theta)
    y = np.sin(theta)
    return x, y

# N.B. Using the next function to deal with df_train_soundscape is
#      not efficient, since there are only 4 distinct longitudes.
def cyclicize_series(series, max_, min_):
    return list(map(lambda number: cyclicize_number(number, max_, min_), series))





