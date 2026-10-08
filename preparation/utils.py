import os
import av
import numpy as np
import scipy.io.wavfile as wavfile
import torch
import torchaudio
import torchvision


def split_file(filename, max_frames=600, fps=25.0):

    lines = open(filename).read().splitlines()

    flag = 0
    stack = []
    res = []

    tmp = 0
    start_timestamp = 0.0

    threshold = max_frames / fps

    for line in lines:
        if "WORD START END ASDSCORE" in line:
            flag = 1
            continue
        if flag:
            word, start, end, score = line.split(" ")
            start, end, score = float(start), float(end), float(score)
            if end < tmp + threshold:
                stack.append(word)
                last_timestamp = end
            else:
                res.append(
                    [
                        " ".join(stack),
                        start_timestamp,
                        last_timestamp,
                        last_timestamp - start_timestamp,
                    ]
                )
                tmp = start
                start_timestamp = start
                stack = [word]
    if stack:
        res.append([" ".join(stack), start_timestamp, end, end - start_timestamp])
    return res


def save_vid_txt(
    dst_vid_filename, dst_txt_filename, trim_video_data, content, video_fps=25
):
    # -- save video
    save2vid(dst_vid_filename, trim_video_data, video_fps)
    # -- save text
    os.makedirs(os.path.dirname(dst_txt_filename), exist_ok=True)
    f = open(dst_txt_filename, "w")
    f.write(f"{content}")
    f.close()


def save_vid_aud(
    dst_vid_filename,
    dst_aud_filename,
    trim_vid_data,
    trim_aud_data,
    video_fps=25,
    audio_sample_rate=16000,
):
    # -- save video
    save2vid(dst_vid_filename, trim_vid_data, video_fps)
    # -- save audio
    save2aud(dst_aud_filename, trim_aud_data, audio_sample_rate)


def save_vid_aud_txt(
    dst_vid_filename,
    dst_aud_filename,
    dst_txt_filename,
    trim_vid_data,
    trim_aud_data,
    content,
    video_fps=25,
    audio_sample_rate=16000,
):
    # -- save video
    if dst_vid_filename is not None:
        save2vid(dst_vid_filename, trim_vid_data, video_fps)
    # -- save audio
    if dst_aud_filename is not None:
        save2aud(dst_aud_filename, trim_aud_data, audio_sample_rate)
    # -- save text
    os.makedirs(os.path.dirname(dst_txt_filename), exist_ok=True)
    f = open(dst_txt_filename, "w")
    f.write(f"{content}")
    f.close()


def save2vid(filename, vid, frames_per_second):
    """
    Saves a uint8 tensor/array of shape (T, H, W, C) or (T, H, W) to video.
    """
    # 1. Ensure output directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # 2. Safely move PyTorch tensors to CPU before converting to NumPy
    if isinstance(vid, torch.Tensor):
        vid = vid.detach().cpu().numpy()

    # 3. Handle grayscale arrays
    if vid.ndim == 3:  # grayscale (T, H, W) -> expand to (T, H, W, C)
        vid = np.stack([vid] * 3, axis=-1)

    container = av.open(filename, mode="w", format="mp4")
    stream = container.add_stream("h264", rate=int(frames_per_second))
    stream.height = int(vid.shape[1])
    stream.width = int(vid.shape[2])
    stream.pix_fmt = "yuv420p"

    for frame_idx in range(vid.shape[0]):
        frame = av.VideoFrame.from_ndarray(vid[frame_idx], format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()


def save2aud(filename, aud, sample_rate):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if isinstance(aud, torch.Tensor):
        aud = aud.detach().cpu().numpy()

    # Ensure 1D audio array for mono output
    aud = np.squeeze(aud)

    # Normalize float waveforms to 16-bit PCM for WAV writing
    if aud.dtype != np.int16:
        aud = (aud * 32767).clip(-32768, 32767).astype(np.int16)

    wavfile.write(filename, sample_rate, aud)
