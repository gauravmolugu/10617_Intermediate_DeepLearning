import pandas as pd
import numpy as np
import pickle

unit_info_path = "stimulus_data/unit_info.csv"
raw_spike_path = "spike_data/raw_spike_data.csv"
save_path = "entire_dict.pkl"

def make_binary_data(spike_df):
    d = dict()
    ids = spike_df["stimulus_presentation_id"].unique()
    index_df = pd.read_csv(unit_info_path, index_col=0)
    index = list(index_df.columns)

    for i in range(len(ids)):
        id = int(ids[i])
        arr = np.zeros((93, 50))
        df = spike_df[spike_df["stimulus_presentation_id"] == id]

        for j in range(df.shape[0]):    # loop through the spikes in each stimulus
            spike = df.iloc[j]
            unit = str(spike["unit_id"])
            time_since = spike["time_since_stimulus_presentation_onset"] * 1000 
            time_index = int(np.floor(time_since)) // 5                         # in milliseconds
            unit_index = index.index(unit)                                      # index of the unit
            if(time_index < 50):
                arr[unit_index, time_index] = 1
        
        d[id] = arr

    return d

def make_spontaneous_data(spike_df):
    d = dict()
    ids = spike_df["stimulus_presentation_id"].unique()
    index_df = pd.read_csv(unit_info_path, index_col=0)
    index = list(index_df.columns)
    prev = None

    for id in ids:
        if(prev != None):
            d.pop(prev)

        df = spike_df[spike_df["stimulus_presentation_id"] == id]
        dict_index = id + 100000
        curr_arr_index = 0
        arr = np.zeros((93, 50))

        for j in range(df.shape[0]):    # loop through the spikes in each stimulus
            spike = df.iloc[j]
            unit = str(spike["unit_id"])
            time_since = spike["time_since_stimulus_presentation_onset"] * 1000
            time_since = time_since - (250 * curr_arr_index)
            time_index = int(np.floor(time_since)) // 5                         # in milliseconds
            unit_index = index.index(unit)                                      # index of the unit
            if(time_index < 50):
                arr[unit_index, time_index] = 1
            else:
                prev = dict_index + curr_arr_index
                d[dict_index + curr_arr_index] = arr
                curr_arr_index += 1
                arr = np.zeros((93, 50))

    return d

df = pd.read_csv(raw_spike_path, index_col=0)

spontaneous_df = df[df["stimulus_name"] == "spontaneous"]       # 0
flashes_df = df[df["stimulus_name"] == "flashes"]               # 1
gratings_df = df[df["stimulus_name"] == "static_gratings"]      # 2
natural_df = df[df["stimulus_name"] == "natural_scenes"]        # 3

flashes_dict = make_binary_data(flashes_df)
gratings_dict = make_binary_data(gratings_df)
natural_dict = make_binary_data(natural_df)
spontaneous_dict = make_spontaneous_data(spontaneous_df)

total_dict = dict()

for id, arr in spontaneous_dict.items():
    total_dict[(id, 0)] = arr

for id, arr in flashes_dict.items():
    total_dict[(id, 1)] = arr

for id, arr in gratings_dict.items():
    total_dict[(id, 2)] = arr

for id, arr in natural_dict.items():
    total_dict[(id, 3)] = arr

with open("entire_dict.pkl", "wb") as f:
    pickle.dump(total_dict, f)