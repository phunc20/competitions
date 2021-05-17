import shutil
from sklearn.model_selection import StratifiedShuffleSplit
from utils import *

df_train_meta = pd.read_csv(PATH_DATASET/"train_metadata.csv")

cols_taken = [
    "primary_label",
    "latitude",
    "longitude",
    "date",
    "filename",
]

df_meta = df_train_meta.loc[:, cols_taken]
df_meta[["year", "month", "day"]] = pd.DataFrame([ date.split("-") for date in df_meta["date"].tolist() ])

L_ogg_paths = (PATH_DATASET / "train_short_audio").glob("*/*.ogg")

# cut audio into pieces of .npy files of 5 sec each and save to tmp/
shortaudio_npy_tmp = Path("shortaudio_npy_tmp")
shortaudio_npy_tmp.mkdir(exist_ok=True)

audios_to_npy(
    L_ogg_paths,
    n_processes=8,
    sr=SR,
    resample=True,
    res_type="kaiser_fast",
    is_soundscape=False,
    step_in_sec=5,
    save_to=shortaudio_npy_tmp,
)

L_npy_filenames = [path.name for path in shortaudio_npy_tmp.glob("*.npy")]
df_xc = pd.DataFrame(np.c_[
    [filename.split("_")[0] + ".ogg" for filename in L_npy_filenames],
     L_npy_filenames,
    ],
    columns=["filename", "npy_filename"]
)
df_join = df_meta.join(df_xc.set_index("filename"), on="filename")
df_join.reset_index(drop=True, inplace=True)


#shortaudio_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
shortaudio_split = StratifiedShuffleSplit(test_size=0.3, random_state=SEED)
for train_indices, val_indices in shortaudio_split.split(df_join, df_join["primary_label"]):
    df_train = df_join.loc[train_indices]
    df_val = df_join.loc[val_indices]

train_npy = Path("train_npy")
train_npy.mkdir(exist_ok=True)
val_npy = Path("val_npy")
val_npy.mkdir(exist_ok=True)

for filename in df_train["npy_filename"]:
    shutil.move(shortaudio_npy_tmp / filename, train_npy / filename)

## TypeError: expected str, bytes or os.PathLike object, not float
for filename in df_val["npy_filename"]:
    shutil.move(shortaudio_npy_tmp / filename, val_npy / filename)

df_train.to_csv("shortaudio_train.csv", index=False)
df_val.to_csv("shortaudio_val.csv", index=False)


