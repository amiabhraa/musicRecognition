import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import maximum_filter
from collections import defaultdict, Counter


#configuration

SONGS_FOLDER = "songs"

SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512

QUERY_DURATION = 10

TARGET_MIN = 10
TARGET_MAX = 100
FAN_OUT = 5


# song fingerprinting

def generate_fingerprints(audio):

    spectrogram = np.abs(
        librosa.stft(
            audio,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
    )

    # Detect local peaks
    local_max = maximum_filter(
        spectrogram,
        size=(7, 7)
    )

    peak_mask = (
        (spectrogram == local_max)
        & (spectrogram > spectrogram.max() * 0.05)
    )

    frequency_bins, time_frames = np.where(peak_mask)

    # Sort peaks chronologically
    peaks = sorted(
        zip(time_frames, frequency_bins),
        key=lambda p: p[0]
    )

    fingerprints = []

    for i, (anchor_time, anchor_freq) in enumerate(peaks):

        targets_found = 0

        for j in range(i + 1, len(peaks)):

            target_time, target_freq = peaks[j]

            delta_time = target_time - anchor_time

            if delta_time < TARGET_MIN:
                continue

            if delta_time > TARGET_MAX:
                break

            fingerprint = (
                int(anchor_freq),
                int(target_freq),
                int(delta_time)
            )

            fingerprints.append(
                (fingerprint, anchor_time)
            )

            targets_found += 1

            if targets_found >= FAN_OUT:
                break

    return fingerprints


# indexing song library

def build_library():

    library = {}

    for filename in os.listdir(SONGS_FOLDER):

        if not filename.lower().endswith(".wav"):
            continue

        path = os.path.join(SONGS_FOLDER, filename)

        print(f"\nIndexing: {filename}")

        audio, sr = librosa.load(
            path,
            sr=SAMPLE_RATE,
            mono=True
        )

        fingerprints = generate_fingerprints(audio)

        song_hashes = defaultdict(list)

        for fingerprint, anchor_time in fingerprints:

            song_hashes[fingerprint].append(anchor_time)

        library[filename] = song_hashes

        print("Fingerprints:", len(fingerprints))

    return library


#query match
def recognize_song(query_audio, library):

    query_fingerprints = generate_fingerprints(query_audio)

    results = []

    for song_name, song_hashes in library.items():

        offset_votes = Counter()
        matched_hashes = 0

        for fingerprint, query_time in query_fingerprints:

            if fingerprint not in song_hashes:
                continue

            matched_hashes += 1

            reference_times = song_hashes[fingerprint]

            for reference_time in reference_times:

                offset = reference_time - query_time

                offset_bin = round(offset / 2) * 2

                offset_votes[offset_bin] += 1

        if offset_votes:

            best_offset, best_votes = offset_votes.most_common(1)[0]

            score = best_votes / max(len(query_fingerprints), 1)

            results.append(
                (
                    song_name,
                    best_votes,
                    score,
                    best_offset
                )
            )

    results.sort(
        key=lambda result: result[1],
        reverse=True
    )

    return results, query_fingerprints


#visual match
def visualize_match(query_audio, song_path, offset_frames):

    # Load reference song
    reference_audio, sr = librosa.load(
        song_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Spectrograms
    reference_spec = np.abs(
        librosa.stft(
            reference_audio,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
    )

    query_spec = np.abs(
        librosa.stft(
            query_audio,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
    )

    reference_db = librosa.amplitude_to_db(
        reference_spec,
        ref=np.max
    )

    query_db = librosa.amplitude_to_db(
        query_spec,
        ref=np.max
    )

    # Detect peaks in reference
    ref_max = maximum_filter(
        reference_spec,
        size=(7, 7)
    )

    ref_mask = (
        (reference_spec == ref_max)
        & (reference_spec > reference_spec.max() * 0.05)
    )

    ref_freq_bins, ref_time_frames = np.where(ref_mask)

    # Detect peaks in query
    query_max = maximum_filter(
        query_spec,
        size=(7, 7)
    )

    query_mask = (
        (query_spec == query_max)
        & (query_spec > query_spec.max() * 0.05)
    )

    query_freq_bins, query_time_frames = np.where(query_mask)

    # Convert to times
    ref_times = librosa.frames_to_time(
        ref_time_frames,
        sr=sr,
        hop_length=HOP_LENGTH
    )

    query_times = librosa.frames_to_time(
        query_time_frames,
        sr=sr,
        hop_length=HOP_LENGTH
    )

    # Convert to frequencies
    ref_frequencies = librosa.fft_frequencies(
        sr=sr,
        n_fft=N_FFT
    )[ref_freq_bins]

    query_frequencies = librosa.fft_frequencies(
        sr=sr,
        n_fft=N_FFT
    )[query_freq_bins]

    # Convert offset frames into seconds
    offset_seconds = (
        offset_frames * HOP_LENGTH / SAMPLE_RATE
    )

    aligned_query_times = query_times + offset_seconds

    #plot
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 10)
    )

    # Reference song spectrogram
    librosa.display.specshow(
        reference_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="log",
        cmap="inferno",
        ax=axes[0]
    )

    axes[0].scatter(
        ref_times,
        ref_frequencies,
        color="orange",
        s=5,
        alpha=0.7,
        label="Reference peaks"
    )

    axes[0].scatter(
        aligned_query_times,
        query_frequencies,
        color="cyan",
        s=8,
        alpha=0.8,
        label="Aligned query peaks"
    )

    axes[0].set_title(
        f"Fingerprint Alignment | Offset: {offset_seconds:.2f}s"
    )

    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].legend()

    # Query spectrogram
    librosa.display.specshow(
        query_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="log",
        cmap="inferno",
        ax=axes[1]
    )

    axes[1].scatter(
        query_times,
        query_frequencies,
        color="cyan",
        s=8,
        label="Query peaks"
    )

    axes[1].set_title("Query Spectrogram")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


#main function
if __name__ == "__main__":

    # Build fingerprint library
    library = build_library()

    if not library:
        print("No WAV songs found in songs folder.")
        exit()

    test_song = "example_song_goes_here.wav"  #text song------------------goes here-----------------------------

    test_path = os.path.join(
        SONGS_FOLDER,
        test_song
    )

    test_audio, sr = librosa.load(
        test_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    start_second = 30

    start_sample = int(start_second * sr)
    end_sample = int((start_second + QUERY_DURATION) * sr)

    query_audio = test_audio[start_sample:end_sample]

    if len(query_audio) == 0:
        print("Query segment is empty.")
        exit()

# recognition
    print("\nRecognizing song...\n")

    results, query_fingerprints = recognize_song(
        query_audio,
        library
    )

    print("Results Terminal")

    if results:

        for rank, result in enumerate(results, start=1):

            song, votes, score, offset = result

            print(f"\nRank: {rank}")
            print(f"Song: {song}")
            print(f"Matching votes: {votes}")
            print(f"Score: {score:.4f}")
            print(f"Estimated offset (frames): {offset}")

        best_song, votes, score, offset = results[0]

        best_song_path = os.path.join(
            SONGS_FOLDER,
            best_song
        )

        visualize_match(
            query_audio,
            best_song_path,
            offset
        )

    else:

        print("No matching song found.")
