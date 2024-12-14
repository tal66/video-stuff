import logging
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.fft import fft2, fftshift
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

MAX_OUTPUT_VIDEO_HEIGHT = 360
COLOR_MAP = cv2.COLORMAP_OCEAN


def frame_at_time(time_str, fps):
    """
    'HH:MM:SS' or 'MM:SS' to frame number
    """
    total_seconds = to_seconds(time_str)
    return int(total_seconds * fps)


def to_seconds(time_str: str) -> int:
    """
    'HH:MM:SS' or 'MM:SS' to seconds
    """
    parts = time_str.split(':')

    if len(parts) == 3:
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        minutes, seconds = map(int, parts)
        return minutes * 60 + seconds
    else:
        raise ValueError(f"invalid time format. expecting 'MM:SS' or 'HH:MM:SS'. got '{time_str}'")


def to_str_hhmmss(seconds: int):
    hours, minutes = divmod(int(seconds), 3600)
    minutes, seconds = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def calculate_dimensions(width, height, max_height=MAX_OUTPUT_VIDEO_HEIGHT):
    """
    calculate new dimensions maintaining aspect ratio with max_height
    """
    if height <= max_height:
        return width, height

    ratio = width / height
    new_height = max_height
    new_width = int(new_height * ratio)
    return new_width, new_height


def process_frame_fourier(frame):
    """apply Fourier transform to frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f_transform = fft2(gray)
    f_shift = fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_shift) + 1)
    # scales values in array to 0-255 and converts to uint8
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(magnitude_spectrum, COLOR_MAP)



def video_to_fourier(input_path, output_path, start_time=None, end_time=None):
    """output video with Fourier transform side by side, and original audio"""

    # rm output file if exists
    if os.path.exists(output_path):
        os.remove(output_path)
        logger.info(f"removed existing output file: {output_path}")

    logger.info(f"starting video processing: {input_path}")

    # load
    video_clip = VideoFileClip(input_path)
    audio = video_clip.audio

    # trim
    if (start_time is not None) or (end_time is not None):
        if (start_time is not None) and (end_time is None):
            duration_sec = video_clip.duration
            end_time = to_str_hhmmss(duration_sec)
        elif (start_time is None) and (end_time is not None):
            start_time = "00:00"

        start_seconds = to_seconds(start_time)
        end_seconds = to_seconds(end_time)
        video_clip = video_clip.subclipped(start_seconds, end_seconds)
        audio = audio.subclipped(start_seconds, end_seconds)

    cap = cv2.VideoCapture(input_path)

    # video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # dim
    frame_width, frame_height = calculate_dimensions(orig_width, orig_height)
    output_height = frame_height
    logger.info(f"Original resolution: {orig_width}x{orig_height}, FPS: {fps}, duration: {total_frames / fps:.0f} sec")
    logger.info(f"Processing resolution: {frame_width}x{frame_height}, duration: {video_clip.duration} sec")

    # frame ranges
    start_frame = 0
    end_frame = total_frames
    if (start_time is not None) and (end_time is not None):
        start_frame = frame_at_time(start_time, fps)
        end_frame = frame_at_time(end_time, fps)

    num_frames = end_frame - start_frame

    # video writer
    temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (frame_width * 2, output_height))

    logger.info("Processing video frames...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    with tqdm(total=num_frames, desc="Processing frames", unit="frame") as pbar:
        current_frame = start_frame
        frame_idx = 0

        while cap.isOpened() and current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:  # indicates whether a frame was successfully read
                break

            # resize
            if orig_height > MAX_OUTPUT_VIDEO_HEIGHT:
                frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)

            fourier_frame = process_frame_fourier(frame)
            final_frame = np.hstack((frame, fourier_frame))

            out.write(final_frame)
            current_frame += 1
            frame_idx += 1
            pbar.update(1)

    # clean
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # add audio to final video
    try:
        add_audio(audio, temp_output, output_path)
        logger.info("audio processing complete")

    except Exception as e:
        logger.exception(e)
        logger.info("Fall back to video without audio")
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(temp_output, output_path)

    video_clip.close()
    os.unlink(temp_output)

    logger.info("processing complete")


def add_audio(audio, video, output_path):
    logger.info("add audio to video...")
    processed_clip = VideoFileClip(video)
    final_clip = processed_clip.with_audio(audio)
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', logger=None)
    logger.info(f"output: {output_path}")
    processed_clip.close()
    final_clip.close()


if __name__ == "__main__":
    input_video = r"video.mp4"
    input_filename = Path(input_video).stem
    output_video = f"./videos/fourier_{input_filename}.mp4"

    # video_to_fourier(input_video, output_video, "0:10", "0:30")
    video_to_fourier(input_video, output_video)
