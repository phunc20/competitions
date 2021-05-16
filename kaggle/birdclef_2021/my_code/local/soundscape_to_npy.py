from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import shutil
from utils import *


df_train_soundscape = pd.read_csv(PATH_DATASET / "train_soundscape_labels.csv")
df_train_soundscape["n_birds"] = list(map(birdLabel_to_nBirds, df_train_soundscape["birds"]))

# Randomly initialize the new columns
df_train_soundscape["year"] = -1
df_train_soundscape["month"] = -1
df_train_soundscape["day"] = -1
df_train_soundscape["longitude"] = 0.0
df_train_soundscape["latitude"] = 0.0
#df_train_soundscape["npy_path"] = None

for p in (PATH_DATASET / "train_soundscapes").iterdir():
    id_, location, date = p.stem.split("_")
    # date, str, "yyyymmdd"
    year = int(date[:4])
    month = int(date[4:6])
    day = int(date[6:])
    id_ = int(id_)
    filter_ = df_train_soundscape.audio_id == id_
    #df_train_soundscape.loc[filter_, ["year"]] = year
    #df_train_soundscape.loc[filter_, ["month"]] = month
    #df_train_soundscape.loc[filter_, ["day"]] = day
    df_train_soundscape.loc[filter_, "year"] = year
    df_train_soundscape.loc[filter_, "month"] = month
    df_train_soundscape.loc[filter_, "day"] = day

for location, coordinate in D_location_coordinate.items():
    lo, la = coordinate.longitude, coordinate.latitude
    location_filter = df_train_soundscape.loc[:, "site"] == location
    df_train_soundscape.loc[location_filter, "longitude"] = lo
    df_train_soundscape.loc[location_filter, "latitude"] = la


df_train_soundscape[["month_x", "month_y"]] = cyclicize_series(df_train_soundscape["month"], 12, 0)

df_train_soundscape[["day_coarse_x", "day_coarse_y"]] = cyclicize_series(df_train_soundscape["day"], 31, 0)

df_train_soundscape[["longitude_x", "longitude_y"]] = cyclicize_series(df_train_soundscape["longitude"], 180, -180)

df_train_soundscape["latitude_normalized"] = df_train_soundscape["latitude"] / 90

soundscape_features = [
        "month_x",
        "month_y",
        "day_coarse_x",
        "day_coarse_y",
        "longitude_x",
        "longitude_y",
        "latitude_normalized",
]

########################################################
##  StratifiedShuffleSplit does not allow value_counts 1
########################################################
##  df_train_soundscape["n_birds"].value_counts()
##  0    1529
##  1     627
##  2     183
##  3      55
##  4       5
##  5       1
##  Name: n_birds, dtype: int64
df_5_birds = df_train_soundscape[df_train_soundscape["n_birds"] == 5]
df_le_4_birds = df_train_soundscape.drop(index=[1974])
df_le_4_birds.reset_index(drop=True, inplace=True)


soundscape_split1 = StratifiedShuffleSplit(test_size=400, random_state=SEED)
for tv_indices, test_indices in soundscape_split1.split(df_le_4_birds, df_le_4_birds["n_birds"]):
    df_soundscape_train_val = df_le_4_birds.loc[tv_indices]
    df_soundscape_test = df_le_4_birds.loc[test_indices]

df_soundscape_train_val.reset_index(drop=True, inplace=True)
#soundscape_split2 = StratifiedShuffleSplit(test_size=400, random_state=SEED)
for train_indices, val_indices in soundscape_split1.split(df_soundscape_train_val, df_soundscape_train_val["n_birds"]):
    df_soundscape_train = df_soundscape_train_val.loc[train_indices]
    df_soundscape_val = df_soundscape_train_val.loc[val_indices]

df_soundscape_train = pd.concat([df_soundscape_train, df_5_birds]) 


## Cut Audios and Placing Them to Train/Val/Test Folders
## - Although by now we have known where to put the cuts, it seems better to cut and save the videos' `.npy` files into a common folder, say `./soundscape_npy_tmp/`, first.
## - Then we shall move each files to its corresponding folder according to `df_soundscape_train/df_soundscape_val/df_soundscape_test`
soundscape_npy_tmp = Path("soundscape_npy_tmp")
soundscape_npy_tmp.mkdir(exist_ok=True)

audios_to_npy(
    list((PATH_DATASET / "train_soundscapes").iterdir()),
    n_processes=4,
    sr=SR,
    resample=True,
    res_type="kaiser_fast",
    is_soundscape=True,
    step_in_sec=5,
    save_to=soundscape_npy_tmp,
)

tmp_train_npy_paths = [ soundscape_npy_tmp / f"{row_id}.npy" for row_id in df_soundscape_train["row_id"] ]
tmp_val_npy_paths = [ soundscape_npy_tmp / f"{row_id}.npy" for row_id in df_soundscape_val["row_id"] ]
tmp_test_npy_paths = [ soundscape_npy_tmp / f"{row_id}.npy" for row_id in df_soundscape_test["row_id"] ]

train_npy = Path("train_npy")
train_npy.mkdir(exist_ok=True)
val_npy = Path("val_npy")
val_npy.mkdir(exist_ok=True)
test_npy = Path("test_npy")
test_npy.mkdir(exist_ok=True)

pool = joblib.Parallel(4)
mv = joblib.delayed(shutil.move)
mv_train_tasks = list(mv(str(path), train_npy)
                      for path in tmp_train_npy_paths)
pool(tqdm(mv_train_tasks))
#mv_val_tasks = list(mv(path, val_npy)
mv_val_tasks = list(mv(str(path), val_npy)
                    for path in tmp_val_npy_paths)
pool(tqdm(mv_val_tasks))
#mv_test_tasks = list(mv(path, test_npy)
mv_test_tasks = list(mv(str(path), test_npy)
                     for path in tmp_test_npy_paths)
pool(tqdm(mv_test_tasks))

df_soundscape_train.to_csv("soundscape_train.csv", index=False)
df_soundscape_val.to_csv("soundscape_val.csv", index=False)
df_soundscape_test.to_csv("soundscape_test.csv", index=False)


####################
## Scripting Area ##
####################
#soundscape_npy_tmp = Path("soundscape_npy_tmp")
#soundscape_npy_tmp.mkdir(exist_ok=True)
#soundscape_npy_tmp.exists()

