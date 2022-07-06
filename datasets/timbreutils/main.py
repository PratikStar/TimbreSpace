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
LOG_LEVEL = 3  # {1-6} High Value = High verbosity


def log(logline, log_level=1):
    global SHOW_LOGS, LOG_LEVEL
    if SHOW_LOGS and log_level <= LOG_LEVEL:
        stack = inspect.stack()
        the_class = stack[1][0].f_locals["self"].__class__.__name__
        the_method = stack[1][0].f_code.co_name
        print("{}.{}: {}".format(the_class, the_method, logline))

class Loader:
    """Loader is responsible for loading an audio file."""

    def __init__(self, sample_rate, load_duration, mono):
        self.sample_rate = sample_rate
        self.load_duration = load_duration
        self.mono = mono

    def load(self, file_path, offset):
        signal = librosa.load(file_path,
                              sr=self.sample_rate,
                              duration=self.load_duration,
                              offset=offset,
                              mono=self.mono)[0]
        log("Shape of the loaded signal: " + str(signal.shape), 2)
        log("Mean of the loaded signal: " + str(np.mean(signal)), 5)
        log("Min of the loaded signal: " + str(np.amin(signal)), 5)
        log("Max of the loaded signal: " + str(np.amax(signal)), 5)
        log("Raw signal: " + str(signal), 6)
        return signal

    # def save(self, filepath, signal):
    #     sf.write(os.path.join(self.audio_segments_save_dir, filepath), signal, self.sample_rate)


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

    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self, signal):
        stft = librosa.stft(signal,
                            n_fft=self.frame_size,
                            hop_length=self.hop_length)[:-1]
        log("Shape of stft: " + str(stft.shape), 2)
        spectrogram = np.abs(stft)  # https://librosa.org/doc/main/generated/librosa.stft.html abs gives the magnitude
        phases = np.angle(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram, phases


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

    def normalise(self, array):
        norm_array = (array - array.min()) / (array.max() - array.min())
        norm_array = norm_array * (self.max - self.min) + self.min
        log("Shape of normalized array: " + str(norm_array.shape), 2)
        log("Max of norm_array: " + str((array.max())), 2)
        log("Min of norm_array: " + str((array.min())), 2)
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


class Visualizer:  # Not used!!
    def __init__(self, file_dir, frame_size, hop_length):
        self.file_dir = file_dir
        self.frame_size = frame_size
        self.hop_length = hop_length

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
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        name = file_name.with_suffix('.png').name
        if suffix is not None:
            name = file_name.name[:file_name.name.index('.wav')] + " - " + suffix + '.png'
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
        global SHOW_LOGS, LOG_LEVEL

        SHOW_LOGS = config.show_logs
        LOG_LEVEL = config.log_level
        self.dataset_path = dataset_path
        self.config = config
        self.padder = None
        self.spectrogram_extractor = None
        self.feature_extractor = None
        self.normaliser = None
        self.saver = None
        self.visualizer = None
        self.reconstructor = None
        self.min_max_values = {}
        self._loader = None
        self._num_expected_samples = None

    @property
    def loader(self):
        return self._loader

    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.load_duration)

    # Processes Single file
    def process_file(self, clip_name, offset, visualize=True):

        file_name_di = self.dataset_path / 'DI.wav'
        file_name = self.dataset_path / 'clips' / clip_name
        log(f"Processing Segment: {file_name}")

        signal = self.loader.load(file_name, offset)
        signal_di = self.loader.load(file_name_di, offset)

        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
        feature, phases = self.spectrogram_extractor.extract(signal)

        if self._is_padding_necessary(signal_di):
            signal_di = self._apply_padding(signal_di)
        feature_di, phases_di = self.spectrogram_extractor.extract(signal_di)

        norm_feature = self.normaliser.normalise(feature)
        log(f"Shape of complete spectrogram: {norm_feature.shape}")

        norm_feature_di = self.normaliser.normalise(feature_di)

        segment_features = []
        segment_features_di = []

        batch_spectrogram_width = math.ceil(
            self.config.batch_size * self.config.load.sample_rate * self.config.stft.segment_duration / self.config.stft.hop_length)
        assert batch_spectrogram_width % self.config.batch_size == 0, \
            f"Width of the batch spectrogram ({batch_spectrogram_width}), is not divisible by batch size ({self.config.batch_size})"

        segment_spectrogram_width = batch_spectrogram_width // self.config.batch_size
        for i in range(0, norm_feature.shape[1], segment_spectrogram_width):
            segment_features.append(norm_feature[:, i: i + segment_spectrogram_width])
            segment_features_di.append(norm_feature_di[:, i: i + segment_spectrogram_width])

        # save_path = self.saver.save_feature(norm_feature, phases, file_name)
        # self.saver.save_min_max_values(file_name, feature.min(), feature.max())

        if visualize:
            self.visualizer.visualize(norm_feature, clip_name, f"{offset:.2f}-batch")
            self.visualizer.visualize(norm_feature_di, 'DI.wav', f"{offset:.2f}-batch")
            for i in range(len(segment_features)):
                self.visualizer.visualize(segment_features[i], clip_name, f"{offset:.2f}-{i}")
                self.visualizer.visualize(segment_features_di[i], 'DI.wav', f"{offset:.2f}-{i}")

        return np.array(segment_features), np.array(segment_features_di)

    def _is_padding_necessary(self, signal):
        if len(signal) < self._num_expected_samples:
            log("Padding necessary", 1)
            log("Actual Signal length: " + str(len(signal)), 5)
            log("Expected samples in signal: " + str(self._num_expected_samples), 5)
            return True
        return False

    def _apply_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal
