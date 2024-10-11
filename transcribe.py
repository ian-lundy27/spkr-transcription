import io
import argparse
import warnings

import numpy as np
from pydub import AudioSegment
import torch
from pyannote.audio import Pipeline
import whisper


class DiarizedSegment:

    def __init__(self, 
                 audio: AudioSegment, 
                 speaker: str,
                 start: int, 
                 end: int,
                 ):
        self.audio: AudioSegment = audio
        self.speaker: str = speaker
        self.start: int = start
        self.end: int = end
        self.content: str = None
        self.nparray: np.ndarray = None

    def diff(self):
        return self.end - self.start


def set_torch_device(device: str | None = None) -> None:

    if not device:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        elif torch.xpu.is_available():
            device = "xpu"
    try:
        torch.set_default_device(device)
    except Exception as e:
        print(f"Failed to set processor to {device}, using cpu by default")
        torch.set_default_device("cpu")


def parse_speakers(audio_buffer: io.BytesIO, auth_token: str) -> list[dict]:

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=auth_token)
    
    if pipeline is None:
        print("Failed to create pipeline from pyannote. Check that auth token is correct and has access to pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0")
        raise Exception()

    pipeline.to(torch.device(torch.get_default_device()))
    diarization = pipeline(audio_buffer)

    timestamps = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        timestamps.append({
            "speaker": speaker,
            "start": round(turn.start * 1000),
            "end": round(turn.end * 1000)
        })

    return timestamps


def join_timestamps(timestamps: list[dict]) -> list[dict]:

    joined_timestamps = []
    speaker = ""
    start = 0
    end = 0

    for timestamp in timestamps:
        if timestamp["speaker"] != speaker:
            joined_timestamps.append({
                "speaker": speaker,
                "start": start,
                "end": end
            })
            speaker = timestamp["speaker"]
            start = timestamp["start"]
        end = timestamp["end"]
    
    joined_timestamps.append({
                "speaker": speaker,
                "start": start,
                "end": end
            })

    return joined_timestamps[1:]


def split_audio_by_timestamp(audio: str | io.BytesIO, timestamps: list[dict]) -> list[DiarizedSegment]:

    audio = AudioSegment.from_file(audio)

    audio_segments = []
    for timestamp in timestamps:
        audio_segments.append(DiarizedSegment(
            audio[timestamp["start"]:timestamp["end"]],
            timestamp["speaker"],
            timestamp["start"],
            timestamp["end"]
        ))
    return audio_segments


def convert_audio_to_array(audio_segment: AudioSegment) -> np.ndarray:
    buffer = io.BytesIO()
    audio_segment.export(buffer, format="s16le", codec="pcm_s16le")
    return np.frombuffer(buffer.getvalue(), np.int16).flatten().astype(np.float32) / 32768.0


def transcribe_segment(loaded_model: whisper.Whisper, audio_segment: DiarizedSegment) -> dict:
    return loaded_model.transcribe(audio_segment.nparray, fp16=False if torch.get_default_device() == "cpu" else True)


def join_transcripts(audio_segments: list[DiarizedSegment]) -> str:
    formatted_segments = []
    for s in audio_segments:
        formatted_segments.append(f"[{s.start}:{s.end}] - {s.speaker}: {s.content}")
    return "\n\n".join(formatted_segments)


def save_to_file(output: str, path: str):
    with open(path, "w") as out:
        out.write(output)


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio", default=None, type=str, help="Path to audio file being transcribed")
    parser.add_argument("auth_token", default=None, type=str, help="Hugging face auth token")
    parser.add_argument("--auth-file", action="store_true", help="Set flag if auth token argument is a file")
    parser.add_argument("--output-file", default="transcript.txt", type=str, help="Path to file where transcript will be saved")
    parser.add_argument("--model", default="turbo", type=str, help="Model of whisper to be used", 
                        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large", "turbo"])
    parser.add_argument("--torch-device", default=None, type=str, help="Which processing unit to use", 
                        choices=["cpu", "cuda", "mps", "xpu"])
    return parser.parse_args()


def read_auth_file(path: str) -> str:
    with open(path, "r") as infile:
        return infile.read()


def convert_to_wav_buffer(audio_path):
    
    buffer = io.BytesIO()
    AudioSegment.from_file(audio_path)[:20000].export(buffer, format="wav")
    return buffer


def main():

    args = cli()
    audio_path = args.audio
    auth_token = read_auth_file(args.auth_token) if args.auth_file else args.auth_token
    output_file = args.output_file
    model_name = args.model
    device = args.torch_device

    set_torch_device(device)

    print(f"Set processor to {torch.get_default_device()}")

    buffer = convert_to_wav_buffer(audio_path)

    print("Converted input to .wav")

    timestamps = join_timestamps(parse_speakers(buffer, auth_token))

    print("Got timestamps")

    audio_segments = split_audio_by_timestamp(buffer, timestamps)

    print("Created audio segments")

    model = whisper.load_model(model_name)

    print(f"Loaded model {model_name}")

    for segment in audio_segments:
        segment.nparray = convert_audio_to_array(segment.audio)
        print("Converted audio to array")
        segment.content = transcribe_segment(model, segment)["text"]

    print("Joining transcripts")

    full_transcript = join_transcripts(audio_segments)

    print("Saving transcript")

    save_to_file(full_transcript, output_file) # make sure everything is utf8 encoded


if __name__=="__main__":
    with warnings.catch_warnings(action="ignore"):
        main()

    '''
    import tqdm
    model = ""
    audio_segments = []
    print("Transcribing segments")
    with tqdm(total=sum(segment.diff() for segment in audio_segments)) as loop:
        for segment in audio_segments:
            segment.nparray = convert_audio_to_array(segment.audio)
            segment.content = transcribe_segment(model, segment)["text"]
            loop.update(segment.diff())
    '''