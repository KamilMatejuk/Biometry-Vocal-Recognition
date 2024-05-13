import wave
from scipy import signal
import numpy as np

def alter_amplitude(sound_file: str, multiplier: float, out_file: str = 'temp.wav') -> str:
    input_wav = wave.open(sound_file, 'r')
    nchannels = input_wav.getnchannels()
    sampwidth = input_wav.getsampwidth()
    framerate = input_wav.getframerate()
    nframes = input_wav.getnframes()

    audio_data = input_wav.readframes(nframes)
    audio_array = np.frombuffer(audio_data, dtype=np.int16) * multiplier

    output_wav = wave.open(out_file, 'w')
    output_wav.setparams((nchannels, sampwidth, framerate, nframes, 'NONE', 'not compressed'))
    output_wav.writeframes(audio_array.tobytes())

    input_wav.close()
    output_wav.close()

    return out_file

def subsample(sound_file: str, amount: int, out_file: str = 'temp.wav') -> str:
    input_wav = wave.open(sound_file, 'r')
    nchannels = input_wav.getnchannels()
    sampwidth = input_wav.getsampwidth()
    framerate = input_wav.getframerate()
    nframes = input_wav.getnframes()

    audio_data = input_wav.readframes(nframes)
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    audio_array = audio_array.reshape((-1, nchannels))
    audio_array_downsampled = signal.decimate(audio_array, amount, axis=0).flatten()

    output_wav = wave.open(out_file, 'w')
    output_wav.setparams((nchannels, sampwidth, framerate // 5, len(audio_array_downsampled), 'NONE', 'not compressed'))
    output_wav.writeframes(audio_array_downsampled.tobytes())

    input_wav.close()
    output_wav.close()

    return out_file

def add_noise(sound_file: str, n_first: int, n_second: int, out_file: str = 'temp.wav') -> str:
    input_wav = wave.open(sound_file, 'r')
    nchannels = input_wav.getnchannels()
    sampwidth = input_wav.getsampwidth()
    framerate = input_wav.getframerate()
    nframes = input_wav.getnframes()

    audio_data = input_wav.readframes(nframes)
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    noise = np.random.normal(n_first, n_second, len(audio_array))
    noisy_audio = audio_array + noise

    max_val = np.iinfo(audio_array.dtype).max
    min_val = np.iinfo(audio_array.dtype).min
    noisy_audio = np.clip(noisy_audio, min_val, max_val)

    noisy_audio_bytes = noisy_audio.astype(np.int16).tobytes()

    output_wav = wave.open(out_file, 'w')
    output_wav.setparams((nchannels, sampwidth, framerate, nframes, 'NONE', 'not compressed'))
    output_wav.writeframes(noisy_audio_bytes)

    input_wav.close()
    output_wav.close()

    return out_file

def adjust_amplitude(noise, target_max_amplitude):
    max_amplitude = np.max(np.abs(noise))
    if max_amplitude > target_max_amplitude / 2:
        factor = target_max_amplitude / (2 * max_amplitude)
        noise = (noise * factor).astype(np.int16)
    return noise

def add_noise_from_file(sound_file: str, noise_file: str, out_file: str = 'temp.wav') -> str:
    input_wav = wave.open(sound_file, 'r')
    input_params = input_wav.getparams()
    input_frames = input_wav.readframes(input_params.nframes)
    input_data = np.frombuffer(input_frames, dtype=np.int16)

    noise_wav = wave.open(noise_file, 'r')
    noise_params = noise_wav.getparams()
    noise_frames = noise_wav.readframes(noise_params.nframes)
    noise_data = np.frombuffer(noise_frames, dtype=np.int16)

    max_amplitude_other = np.max(np.abs(input_data))
    noise_data_adjusted = adjust_amplitude(noise_data.copy(), max_amplitude_other / 2)

    if len(noise_data_adjusted) > len(input_data):
        noise_data_adjusted = noise_data_adjusted[:len(input_data)]

    result_data = input_data + noise_data_adjusted

    max_val = np.iinfo(input_data.dtype).max
    min_val = np.iinfo(input_data.dtype).min
    result_data = np.clip(result_data, min_val, max_val)

    output_wav = wave.open(out_file, 'w')
    output_wav.setparams(input_params)
    output_wav.writeframes(result_data.tobytes())

    input_wav.close()
    noise_wav.close()
    output_wav.close()

    return out_file