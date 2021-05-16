from sklearn.model_selection import train_test_split
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





