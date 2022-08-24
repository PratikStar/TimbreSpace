"""
1- load a file..
2- pad the signal (if necessary)
3- extracting log spectrogram from signal
4- normalise spectrogram
5- save the normalised spectrogram

PreprocessingPipeline
"""
import os
import pickle
import re
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import math
import sys
import inspect

SHOW_LOGS = False
LOG_LEVEL = 6  # {1-6} High Value = High verbosity


def log(logline, log_level=1):
    global SHOW_LOGS, LOG_LEVEL
    if SHOW_LOGS and log_level <= LOG_LEVEL:
        stack = inspect.stack()
        the_class = stack[1][0].f_locals["self"].__class__.__name__
        the_method = stack[1][0].f_code.co_name
        print("{}.{}: {}".format(the_class, the_method, logline))


class Loader:
    """Loader is responsible for loading an audio file."""

    def __init__(self, sample_rate, mono):
        self.sample_rate = sample_rate
        self.mono = mono

    def load(self, file_path, offset, load_duration=None):
        signal = librosa.load(file_path,
                              sr=self.sample_rate,
                              duration=load_duration,
                              offset=offset,
                              mono=self.mono)[0]
        log("Shape of the loaded signal: " + str(signal.shape), 2)
        return signal


class Padder:
    """Padder is responsible to apply padding to an array."""

    def __init__(self, mode="constant"):
        self.mode = mode

    def left_pad(self, array, num_missing_items):
        log("Applying Left Padding: " + str(num_missing_items), 5)
        padded_array = np.pad(array,
                              (num_missing_items, 0),
                              mode=self.mode)
        return padded_array

    def right_pad(self, array, num_missing_items):
        log("Applying Right Padding: " + str(num_missing_items), 5)
        padded_array = np.pad(array,
                              (0, num_missing_items),
                              mode=self.mode)
        return padded_array


class LogSpectrogramExtractor:
    """LogSpectrogramExtractor extracts log spectrograms (in dB) from a
    time-series signal.
    """

    def __init__(self, frame_size, hop_length, config):
        self.spectrogram_config = config
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self, signal):
        if self.spectrogram_config.type == "stft":
            stft = librosa.stft(signal,
                                n_fft=self.frame_size,
                                hop_length=self.hop_length)[:-1]
            spectrogram = np.abs(
                stft)  # https://librosa.org/doc/main/generated/librosa.stft.html abs gives the magnitude
            phases = np.angle(stft)
            log_spectrogram = librosa.amplitude_to_db(spectrogram)
            return log_spectrogram, phases
        elif self.spectrogram_config.type == "mel":
            mel = librosa.feature.melspectrogram(y=signal,
                                                 n_fft=self.frame_size,
                                                 hop_length=self.hop_length,
                                                 n_mels=self.spectrogram_config.mel.spectrogram_dims[0])
            return mel, None


class FeatureExtractor:  # Not used!!
    """ Extracts some features like Clip ID and Passage ID"""

    # Clip ID is same as AmpID
    def extract_clipid_from_name(self, filename):
        clip_id = int(filename.split('-')[0])
        log("Clip ID: " + str(clip_id), 5)
        return clip_id

    # Passage ID is equivalent to what used to be 'subclip' ID
    def extract_passageid_from_name(self, filename):
        passage_id = int(filename.split('-')[1].split(' ')[0])
        log("Passage ID: " + str(passage_id), 5)
        return passage_id


class MinMaxNormaliser:
    """MinMaxNormaliser applies min max normalisation to an array."""

    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

    def normalise(self, array, min_ref, max_ref):
        norm_array = (array - min_ref) / (max_ref - min_ref)
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array

    def denormalise(self, norm_array, original_min, original_max):
        array = (norm_array - self.min) / (self.max - self.min)
        array = array * (original_max - original_min) + original_min
        return array


class Saver:  # Not used!!
    """saver is responsible to save features, and the min max values."""

    def __init__(self, feature_save_dir, min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        Utils._create_folder_if_it_doesnt_exist(self.feature_save_dir)
        self.min_max_values_save_dir = min_max_values_save_dir
        Utils._create_folder_if_it_doesnt_exist(self.min_max_values_save_dir)

    def save_feature(self, feature, phases, file_name):
        save_path = self._generate_save_path(file_name)
        save_phases_path = self._generate_save_phases_path(file_name)
        np.save(save_path, feature)
        np.save(save_phases_path, phases)
        return save_path

    def save_min_max_values(self, file_name, min_value, max_value):
        pkl_file = os.path.join(self.min_max_values_save_dir,
                                "min_max_values.pkl")
        data = {}
        if os.path.exists(pkl_file) and os.path.getsize(pkl_file) > 0:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
        data[file_name] = {
            "min": min_value,
            "max": max_value
        }
        with open(pkl_file, 'wb') as f:
            pickle.dump(data, f)

    def _generate_save_path(self, file_name):
        save_path = file_name.parents[1] / file_name.with_suffix(".npy").name
        return save_path

    def _generate_save_phases_path(self, file_name):
        save_path = file_name.parents[1] / file_name.with_suffix(".npy").name
        return save_path


class Visualizer:
    def __init__(self, file_dir, frame_size, hop_length, config):
        self.file_dir = file_dir
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.config = config

    def visualize(self, spectrogram, file_name, suffix=None):
        file_name = self.file_dir / file_name
        plt.ioff()
        fig, ax = plt.subplots(dpi=120)
        try:
            img = librosa.display.specshow(spectrogram,
                                           n_fft=self.frame_size,
                                           hop_length=self.hop_length,
                                           y_axis='log', x_axis='s', ax=ax)
        except IndexError as e:
            log("Null spectrogram for file: " + file_name.name, 1)
            return
        ax.set_title(
            "Frame Size: {}, Hop length: {}, \n{}".format(self.frame_size, self.hop_length,
                                                          file_name.name))
        fig.colorbar(img, ax=ax, format="%+2.2f dB")
        name = file_name.with_suffix('.png').name
        if suffix is not None:
            name = file_name.name[:file_name.name.index('.wav')] + " - " + suffix + '.png'
        fig.savefig(file_name.parents[0] / name)
        plt.close(fig)

    def visualize_multiple(self, spectrograms, suffix=None, file_dir=None, max_rows=5, col_titles=[], title="",
                           filename=None):
        # print("in visualize mul")
        if file_dir is not None and not file_dir.exists():
            os.makedirs(file_dir)

        # print(batch_di.shape)
        # print(di_recons.shape)
        file_name = 'reconstruction.png' if filename is None else filename
        file_name = (file_dir if file_dir is not None else self.file_dir) / file_name
        plt.ioff()
        shape = spectrograms[0].shape
        nrows = min(max_rows, shape[0])
        ncols = len(spectrograms)
        fig, ax = plt.subplots(dpi=120, nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 3))

        for i in range(nrows):
            # print(i)
            for j in range(ncols):
                try:
                    if self.config.spectrogram.type == "stft":
                        img = librosa.display.specshow(spectrograms[j][i, :, :],
                                                       n_fft=self.frame_size,
                                                       hop_length=self.hop_length,
                                                       y_axis='log',
                                                       x_axis='s', ax=ax[i][j])
                    else:
                        img = librosa.display.specshow(spectrograms[j][i, :, :],
                                                       n_fft=self.frame_size,
                                                       hop_length=self.hop_length,
                                                       x_axis='time',
                                                       y_axis='mel',
                                                       sr=self.config.load.sample_rate,
                                                       ax=ax[i][j])
                    # img_reamped = librosa.display.specshow(batch[i, :, :],
                    #                                   n_fft=self.frame_size,
                    #                                   hop_length=self.hop_length,
                    #                                   y_axis='log', x_axis='s', ax=ax[i][0])
                    # img_di = librosa.display.specshow(batch_di[i, :, :],
                    #                                   n_fft=self.frame_size,
                    #                                   hop_length=self.hop_length,
                    #                                   y_axis='log', x_axis='s', ax=ax[i][1])
                    # img_recons = librosa.display.specshow(di_recons[i, :, :],
                    #                                       n_fft=self.frame_size,
                    #                                       hop_length=self.hop_length,
                    #                                       y_axis='log', x_axis='s', ax=ax[i][2])
                except IndexError as e:
                    log("Null spectrogram for file: " + file_name.name, 1)
                    return
                fig.colorbar(img, ax=ax[i][j], format="%+2.2f dB")

        for j in range(len(col_titles)):
            ax[0][j].set_title(col_titles[j])
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.suptitle(title)
        name = file_name.name
        if suffix is not None:
            name = file_name.name[:file_name.name.index('.png')] + " - " + suffix + '.png'
        fig.savefig(file_name.parents[0] / name)
        plt.close(fig)


class Utils:
    def __init__(self):
        pass

    @staticmethod
    def _create_folder_if_it_doesnt_exist(folder):
        if not os.path.exists(folder):
            os.makedirs(folder)


class AudioReconstructor:  ## Not used!!
    """Processes stft inverse and saves the audio file"""

    def __init__(self, file_dir, hop_length, frame_size):
        self.file_dir = file_dir
        self.hop_length = hop_length
        self.frame_size = frame_size
        Utils._create_folder_if_it_doesnt_exist(self.file_dir)

    def reconstruct(self, features, file_name):
        # Invert using Griffin-Lim
        log("Shape of features: " + str(features.shape), 1)
        features_inv = librosa.griffinlim(features,
                                          hop_length=self.hop_length,
                                          win_length=self.frame_size)
        log("Shape of Inverse features: " + str(features_inv.shape), 1)
        Utils._create_folder_if_it_doesnt_exist(self.file_dir)
        sf.write(os.path.join(self.file_dir, file_name), features_inv, 22050)

    def reconstruct_tf_istft(self, features, file_name):
        # Invert using tf.signal.inverse_stft TODO
        # save audio
        pass

    def reconstruct_from_path(self, file_path):
        # Invert using Griffin-Lim
        features = np.load(file_path)
        self.reconstruct(features, os.path.basename(file_path))


class PreprocessingPipeline:
    """PreprocessingPipeline processes a single audio file, applying
    the following steps to each file:
        1- load a file
        2- pad the signal (if necessary)
        3- extracting log spectrogram from signal
        4- normalise spectrogram
        5- save the normalised spectrogram

    Storing the min max values for all the log spectrograms.
    """

    def __init__(self, dataset_path, config):
        self.dataset_path = dataset_path
        self.config = config
        self.loader = None
        self.padder = None
        self.spectrogram_extractor = None
        self.feature_extractor = None
        self.normaliser = None
        self.saver = None
        self.visualizer = None
        self.reconstructor = None
        self.min_max_values = {}

    # Processes Single file
    # this method loads the clip for the batch duration --> converts to spectrogram --> splits the spectrogram
    def process_file(self, clip_name, offset, visualize=False):

        file_name_di = self.dataset_path / 'DI.wav'
        file_name = self.dataset_path / 'clips' / clip_name
        log(f"Processing Segment: {file_name}")

        # min_reamp, max_reamp = self.get_batch_min_max(file_name, offset, self.config.batch_duration)
        # min_di, max_di = self.get_batch_min_max(file_name_di, offset, self.config.batch_duration)
        min_ref = math.inf
        max_ref = math.inf * -1

        segment_features = []
        segment_features_di = []
        segment_signal = []
        segment_signal_di = []

        for batch in range(self.config.batch_size):

            signal = self.loader.load(file_name, offset, self.config.load_duration)
            signal_di = self.loader.load(file_name_di, offset, self.config.load_duration)

            feature, phases = self.spectrogram_extractor.extract(signal)
            feature_di, phases_di = self.spectrogram_extractor.extract(signal_di)

            min_ref = feature.min() if feature.min() < min_ref else min_ref
            min_ref = feature_di.min() if feature_di.min() < min_ref else min_ref

            max_ref = feature.max() if feature.max() > max_ref else max_ref
            max_ref = feature_di.max() if feature_di.max() > max_ref else max_ref

            segment_features.append(feature)
            segment_features_di.append(feature_di)
            segment_signal.append(signal)
            segment_signal_di.append(signal_di)

            offset += self.config.load_duration

        for i in range(self.config.batch_size):
            feature = segment_features[i]
            norm_feature = self.normaliser.normalise(feature, min_ref, max_ref)
            segment_features[i] = [norm_feature]

            feature_di = segment_features_di[i]
            norm_feature_di = self.normaliser.normalise(feature_di, min_ref, max_ref)
            segment_features_di[i] = [norm_feature_di]

            assert norm_feature.min() >= 0
            assert norm_feature.max() <= 1
            assert norm_feature_di.min() >= 0
            assert norm_feature_di.max() <= 1




        if visualize:
            self.visualizer.visualize(np.concatenate(segment_features, axis=2), clip_name, f"{offset:.2f}-batch")
            self.visualizer.visualize(np.concatenate(segment_features_di, axis=2), 'DI.wav', f"{offset:.2f}-batch")
            for i in range(len(segment_features)):
                self.visualizer.visualize(segment_features[i][0], clip_name, f"{offset:.2f}-{i}")
                self.visualizer.visualize(segment_features_di[i][0], 'DI.wav', f"{offset:.2f}-{i}")

        return np.array(segment_features), np.array(segment_features_di), np.array(segment_signal), np.array(
            segment_signal_di), min_ref, max_ref

