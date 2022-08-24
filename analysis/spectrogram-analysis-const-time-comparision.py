import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('agg')
# batches64 = [0, 3, 13, 48, 51, 55, 73, 74, 75, 76, 80, 97, 98, 99, 104, 112, 115, 117, 125]


sr = 22050

frame_sizes = [2048, 1024, 512, 256]
hop_length_divisors = [2, 4, 8]
basepath = "../../spectrogram-analysis-out/const-time-comp"
offsets = [0.37, 88.79, 0]
load_durations = [1, 2, 5] # sec

clips = ["01A US Double Nrm.wav", "05C Placater Dirty.wav"]

def get_spectrogram(path,
                    load_duration,
                    offset,
                    frame_size,
                    hop_length,
                    ):
    global sr
    y, sr = librosa.load(path,
                         sr=sr,
                         duration=load_duration,
                         offset=offset,
                         mono=True)

    spectrogram = librosa.stft(y,
                               n_fft=frame_size,
                               hop_length=hop_length)

    magnitude_spectrogram = np.abs(spectrogram)
    db_spectrogram = librosa.amplitude_to_db(magnitude_spectrogram)

    if db_spectrogram.max() - db_spectrogram.min() == 0:
        print(f"Skipping null spectrogram at offset: {offset}")
        return None

    norm_db_spectrogram = (db_spectrogram - db_spectrogram.min()) / (db_spectrogram.max() - db_spectrogram.min())
    return norm_db_spectrogram

for load_duration in load_durations:
    print(f"load duration: {load_duration}")
    for offset in offsets:
        print(f"Offset: {offset}")

        path = f"{basepath}/DI-{'-'.join([clip.split()[0] for clip in clips])}/load_duration_{load_duration}/offset_{offset}"
        if not os.path.exists(f"{path}"):
            print(f"Creating dir: {path}")
            os.makedirs(f"{path}")

        for frame_size in frame_sizes:
            for hop_length_divisor in hop_length_divisors:

                hop_length = frame_size // hop_length_divisor

                if offset + load_duration > 101:
                    break

                plt.ioff()
                fig, ax = plt.subplots(dpi=120, ncols=1 + len(clips), figsize=(12, 3), tight_layout=True)



                norm_db_spectrogram = get_spectrogram("/Users/pratik/data/timbre/DI.wav",
                                                      load_duration=load_duration,
                                                      offset=offset,
                                                      frame_size=frame_size,
                                                      hop_length=hop_length)
                img = librosa.display.specshow(norm_db_spectrogram,
                                               n_fft=frame_size,
                                               hop_length=hop_length,
                                               y_axis='log', x_axis='s', ax=ax[0])
                spectrogram_width = norm_db_spectrogram.shape[1]
                ax[0].set_title("DI.wav")
                fig.colorbar(img, ax=ax[0], format="%+2.2f dB")

                for i in range(len(clips)):
                    norm_db_spectrogram = get_spectrogram(f"/Users/pratik/data/timbre/clips/{clips[i]}",
                                                          load_duration=load_duration,
                                                          offset=offset,
                                                          frame_size=frame_size,
                                                          hop_length=hop_length)

                    img = librosa.display.specshow(norm_db_spectrogram,
                                                   n_fft=frame_size,
                                                   hop_length=hop_length,
                                                   y_axis='log', x_axis='s', ax=ax[i+1])

                    ax[i+1].set_title(clips[i])
                    fig.colorbar(img, ax=ax[i+1], format="%+2.2f dB")

                suptitle = f"n_fft-{frame_size}, hop_len-{hop_length}, spec_width-{spectrogram_width}"

                fig.suptitle(suptitle, fontsize=14)
                fig.savefig(f"{path}/{suptitle}.png")
                plt.close(fig)
