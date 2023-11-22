"""
Quick experiment to extract single wave cycles from a short audio file of a single (complex) note
and write each to a new WAV file
"""
import os.path
import sys

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.signal import find_peaks
import soundfile as sf


def autocorr(x):
    no_dc_x = (x - np.mean(x))  # remove DC offset --> zero mean signal
    acorr = np.correlate(no_dc_x, no_dc_x, mode='full')  # full autocorrelation function (ACF)
    norm_acorr = (acorr / np.max(abs(acorr)))[-len(x):]  # normalize & grab second half of ACF

    # Index 0 corresponds to 0 delay between the two copies of the signal,
    # i.e. 0 lag is always the max of the ACF function
    assert norm_acorr[0] == 1.0
    assert norm_acorr.argmax() == 0
    return norm_acorr


def plot_peaks_01(x, norm_acorr):
    # find peaks
    inflection = np.diff(np.sign(np.diff(norm_acorr)))  # Find the second-order differences

    # debug
    max_plot_samples = 5000
    # plotwav(np.diff(norm_acorr[0:max_plot_samples]), savefig_name="03_first_diff.png")
    # plotwav(np.diff(np.sign(norm_acorr[0:max_plot_samples])), savefig_name="04_first_diff_sign.png")
    # plotwav(inflection[0:max_plot_samples], savefig_name="05_inflection_2nd_order_diff.png")

    # Find where they are negative (postive peaks in ACF), advance index +1
    peaks = (inflection < 0).nonzero()[0] + 1
    best_peaks = peaks[norm_acorr[peaks] > 0.95]  # Of those, find the index with the maximum value
    print("best_peaks: {} ".format(best_peaks))

    x = x[:max_plot_samples]
    t = np.arange(0, max_plot_samples)
    peak_indices = best_peaks[(best_peaks < max_plot_samples)]
    print(peak_indices, norm_acorr[peak_indices])
    plt.plot(t, x)
    plt.scatter(t[peak_indices], x[peak_indices], c='r')
    plt.show()


def plot_peaks_02(x, norm_acorr, max_plot_samples=10000):
    x = x[:max_plot_samples]
    norm_acorr = norm_acorr[:max_plot_samples]
    peaks, _ = find_peaks(norm_acorr, width=20)
    peaks2, _ = find_peaks(norm_acorr, prominence=1)  # BEST!
    print("Found peaks2 @ the following sample offsets: {}".format(peaks2))

    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    plt.title("Autocorrelation peaks (via width)")
    plt.plot(peaks, norm_acorr[peaks], "xr")
    plt.plot(norm_acorr)
    plt.legend(['width'])

    plt.subplot(2, 2, 2)
    plt.title("Autocorrelation peaks (via prominence)")
    plt.plot(peaks2, norm_acorr[peaks2], "or")
    plt.plot(norm_acorr)
    plt.legend(['prominence'])

    plt.subplot(2, 2, 3)
    plt.title("Original waveform with dots at single-cycle points")
    plt.plot(peaks, x[peaks], "xr")
    plt.plot(x)

    plt.subplot(2, 2, 4)
    plt.title("Original waveform with dots at single-cycle points")
    plt.plot(peaks2, x[peaks2], "or")
    plt.plot(x)
    plt.tight_layout()

    plt.show()


def plot_wav(audio_buffer, savefig_name=None):
    sns.set()  # Use seaborn's default style to make attractive graphs
    plt.rcParams['figure.dpi'] = 100  # Show nicely large images in this notebook
    plt.figure(figsize=(10, 8))

    # if stereo, grab left
    if audio_buffer.ndim != 1:
        audio_buffer = audio_buffer[0, :]

    t = np.arange(0, len(audio_buffer))
    plt.plot(t, audio_buffer)
    plt.xlim([np.min(t), np.max(t)])
    plt.xlabel("time [samples]")
    plt.ylabel("amplitude")
    if savefig_name:
        plt.tight_layout()
        plt.savefig(savefig_name)  # or plt.savefig("sound.pdf")
    else:
        plt.show()


if __name__ == '__main__':
    input_audio_path = sys.argv[1]
    sig, sr = librosa.load(input_audio_path, sr=None, mono=False)
    print("sample rate: {}".format(sr))

    # If stereo grab left
    if sig.shape[0] != 1:
        sig = sig[0, :]

    acf = autocorr(sig)
    # plot_peaks_01(sig, acf)
    plot_peaks_02(sig, acf)

    # Use prominence keyword argument
    acf_peaks, _ = find_peaks(acf, prominence=1)
    print("# acf_peaks: {}, acf_peaks: {}".format(len(acf_peaks), acf_peaks))

    # Snip original file at acf_autocorr_function peak points, which point are
    # principal periodicities in the original wave
    start_index = 0
    basename, ext = os.path.splitext(os.path.basename(input_audio_path))
    for end_index in acf_peaks:
        single_cycle_file_name = basename + "_" + str(start_index).zfill(5) + "_" + str(end_index).zfill(5) + ext
        sf.write(single_cycle_file_name,
                 sig[start_index:end_index],
                 sr,
                 'PCM_16')
        print("wrote file => {}  of length: {} samples ".format(single_cycle_file_name, end_index - start_index))
        start_index = end_index