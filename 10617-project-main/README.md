# 10417/10617 Intermediate Deep Learning Project

The goal of this project is to better represent time-series high-dimensional neural activity data in
lower dimensions. Through this, we hope to learn more about how neural activity is related to how
our brain functions, which is still a topic of great research and importance. We used data collected by
the Allen Institute Neuropixels Project to measure mice brain neural activity after presenting a
visual stimulus, such as an image or video.

Some of the data is too large to be included inside the repository. They can be accessed [in Google Drive here](https://drive.google.com/drive/folders/1rgIwpdIgB7QQMDKXomek918gI49joWZF?usp=sharing) if you have a CMU account.
The only data files required to run the autoencoder and transformer are the `trimmed_data.csv` and `entire_dict.pkl` file, which are used by `autoencoder.ipynb` and `transformer.ipynb`.

The other files are raw data files which we modified and reshaped to get `trimmed_data.csv` and `entire_dict.pkl` and the Python scripts we used to perform that.
