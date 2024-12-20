import logging
import os
from pathlib import Path

import librosa
import soundfile as sf
from moviepy import Clip, AudioFileClip
from moviepy.video.fx import MultiplySpeed
from moviepy.video.io.VideoFileClip import VideoFileClip

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def adjust_video_speed(input_path, speed_factor=1.1):
    """
    change video speed without messing up the pitch.
    not efficient but works.
    """
    if speed_factor <= 0 or speed_factor == 1:
        raise ValueError("speed_factor must be > 0 and != 1")

    input_filename = Path(input_path).stem
    prefix = "slow" if speed_factor < 1 else "fast"
    output_path = f"{prefix}_{input_filename}.mp4"

    try:
        video = VideoFileClip(input_path)
        logger.info(f"input size: {os.path.getsize(input_path) / 1e6:.2f}MB, duration: {video.duration:.2f}s")

        # new_duration = video.duration / speed_factor

        # adjust speed
        # new_video: Clip = video.with_fps(video.fps * speed_factor)
        new_video: Clip = MultiplySpeed(speed_factor).apply(video)

        # audio
        extract_audio = video.audio
        temp_audio_file = "temp_audio.wav"
        extract_audio.write_audiofile(temp_audio_file, fps=extract_audio.fps)
        y, sr = librosa.load(temp_audio_file, mono=True)
        y_speed = librosa.effects.time_stretch(y, rate=speed_factor)
        sf.write(temp_audio_file, y_speed, sr)
        audio_clip = AudioFileClip(temp_audio_file)
        new_video = new_video.with_audio(audio_clip)  # ignore warning

        # save
        logger.info(f"saving video to: {output_path}")
        new_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            preset='medium',
        )

        logger.info(f"new duration: {new_video.duration:.2f}s, new size: {os.path.getsize(output_path) / 1e6:.2f}MB")

        # clean
        video.close()
        new_video.close()
        os.remove(temp_audio_file)

        return output_path

    except Exception as e:
        logger.exception(e)
        raise


if __name__ == "__main__":
    input_video = "input.mp4"

    adjust_video_speed(input_video, speed_factor=1.05)
