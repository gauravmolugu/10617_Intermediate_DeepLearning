import pandas as pd

unit_info_path = "stimulus_data/unit_info.csv"
stimulus_info_path = "stimulus_data/stimulus_info.csv"
save_path = "stimulus_data/data.csv"

def change_categories(label):
    if(label == "spontaneous"):
        return 0
    elif(label == "flashes"):
        return 1
    elif(label == "static_gratings"):
        return 2
    else:
        assert(label == "natural_scenes")
        return 3

units_df = pd.read_csv(unit_info_path)
stimulus_df = pd.read_csv(stimulus_info_path)
stimulus = stimulus_df["stimulus_name"]
df = pd.merge(stimulus_df, units_df, "inner", "stimulus_presentation_id")

names = df["stimulus_name"].to_numpy()
keep = names == "spontaneous"
keep += names == "flashes"
keep += names == "static_gratings"
keep += names == "natural_scenes"

df = df.iloc[keep]
keep = df["duration"].to_numpy() < 0.5
df = df.iloc[keep]

df["stimulus_name"] = df["stimulus_name"].apply(change_categories)
df.to_csv(save_path)