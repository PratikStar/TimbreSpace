import os
import pickle
import re
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import soundfile as sf
import math
import sys
import inspect
matplotlib.use('agg')
# batches64 = [0, 3, 13, 48, 51, 55, 73, 74, 75, 76, 80, 97, 98, 99, 104, 112, 115, 117, 125]


sr = 22050
# frame_size = 256
# hop_length = frame_size // 8
# spectrogram_width = 64

frame_sizes = [2048, 1024, 512, 256]
hop_length_divisors = [2, 4, 8]
basepath = "../../spectrogram-analysis-out/const-time"
offsets = [0.37, 88.79]
load_durations = [1, 2] # sec

for load_duration in load_durations:
    print(f"load duration: {load_duration}")
    for offset in offsets:
        print(f"Offset: {offset}")

        path = f"{basepath}/load_duration_{load_duration}/offset_{offset}"

        if not os.path.exists(f"{path}"):
            print(f"Creating dir: {path}")
            os.makedirs(f"{path}")

        for frame_size in frame_sizes:
            for hop_length_divisor in hop_length_divisors:

                hop_length = frame_size // hop_length_divisor
                # spectrogram_width = load_duration * sr / hop_length

                if offset + load_duration > 101:
                    break
                y, sr = librosa.load("/Users/pratik/data/timbre/DI.wav",
                                     sr=sr,
                                     duration=load_duration,
                                     offset=offset,
                                     mono=True)

                spectrogram = librosa.stft(y,
                                           n_fft=frame_size,
                                           hop_length=hop_length)

                magnitude_spectrogram = np.abs(spectrogram)
                db_spectrogram = librosa.amplitude_to_db(magnitude_spectrogram)

                spectrogram_width = db_spectrogram.shape[1]
                # filename = f"out-{spectrogram_width}-{frame_size}-{hop_length}"

                if db_spectrogram.max() - db_spectrogram.min() == 0:
                    print(f"Skipping null spectrogram at offset: {offset}")
                    continue
                norm_db_spectrogram = (db_spectrogram - db_spectrogram.min()) / (db_spectrogram.max() - db_spectrogram.min())

                plt.ioff()
                fig, ax = plt.subplots(dpi=120)
                img = librosa.display.specshow(norm_db_spectrogram,
                                               n_fft=frame_size,
                                               hop_length=hop_length,
                                               y_axis='log', x_axis='s', ax=ax)
                title = f"n_fft-{frame_size} hop_l-{hop_length} spec_width-{spectrogram_width}"
                ax.set_title(title)
                fig.colorbar(img, ax=ax, format="%+2.2f dB")
                fig.savefig(f"{path}/{title}.png")
                plt.close(fig)
