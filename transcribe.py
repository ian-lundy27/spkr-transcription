import io
import argparse
import logging
import warnings
from pathlib import Path

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
        self.content: str = ""

    def diff(self) -> int:
        return self.end - self.start


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(usage="python transcribe.py <audio file> <auth token> <args>", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.usage
    parser.add_argument("audio", default=None, type=str, help="Path to audio file being transcribed")
    parser.add_argument("auth_token", default=None, type=str, help="Hugging face auth token")
    parser.add_argument("--auth-file", action="store_true", help="Set flag if auth token argument is a file")
    parser.add_argument("--output-file", default="transcript.txt", type=str, help="Path to file where transcript will be saved")
    parser.add_argument("--model", default="turbo", type=str, help="Model of whisper to be used", 
                        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large", "turbo"])
    parser.add_argument("--torch-device", default=None, type=str, help="Which processing unit to use", 
                        choices=["cpu", "cuda", "mps", "xpu"])
    parser.add_argument("--language", default=None, type=str, help="Language to transcribe audio in")
    parser.add_argument("-q", "--quiet", help="Mute stdout", action="store_const", dest="loglevel", const=logging.CRITICAL, default=logging.INFO)
    return parser.parse_args()


def set_torch_device(device: str | None = None) -> None:
    if not device:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
    try:
        torch.set_default_device(device)
    except Exception as e:
        logging.warning(f"Failed to set device to {device}, using cpu by default")
        torch.set_default_device("cpu")


def read_auth_file(path: Path) -> str:
    with open(path, "r") as infile:
        return infile.read().strip()


def parse_speakers(audio: AudioSegment, auth_token: str) -> list[dict[str, str | int]]:
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=auth_token)
    
    if pipeline is None:
        logging.critical("Failed to create pipeline from pyannote. Check that auth token is correct and has access to pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0")
        raise Exception()

    pipeline.to(torch.device(torch.get_default_device()))
    diarization = pipeline(buffer)

    timestamps = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        timestamps.append({
            "speaker": speaker,
            "start": round(turn.start * 1000),
            "end": round(turn.end * 1000)
        })

    return timestamps


def join_timestamps(timestamps: list[dict[str, str | int]], audio: AudioSegment) -> list[DiarizedSegment]:
    joined_timestamps = []
    speaker = ""
    start = 0
    end = 0

    for timestamp in timestamps:
        if timestamp["speaker"] != speaker:
            joined_timestamps.append(DiarizedSegment(
                audio[start:end],
                speaker,
                start,
                end
            ))
            speaker = timestamp["speaker"]
            start = timestamp["start"]
        end = timestamp["end"]
    
    joined_timestamps.append(DiarizedSegment(
        audio[start:end],
        speaker,
        start,
        end
    ))

    return joined_timestamps[1:]


def join_transcripts(speaker_segments: list[DiarizedSegment]) -> str:
    formatted_segments = []
    maxtime = speaker_segments[-1].end
    for s in speaker_segments:
        formatted_segments.append(f"[{create_timestamp(s.start,maxtime)}-{create_timestamp(s.end,maxtime)}] - {s.speaker}: {s.content}")
    return "\n\n".join(formatted_segments)


def create_timestamp(ms: int, maxtime: int) -> str:
    time = ms // 1000
    stamp = f"{time % 3600 // 60:02}:{time % 60:02}"
    if maxtime >= 3600000:
        stamp = f"{time % 21600 // 3600:02}:" + stamp
    return stamp


def save_to_file(output: str, path: Path) -> None:
    with open(path, "wb") as out:
        out.write(output.encode("utf-8"))


def transcribe_segment(segment: DiarizedSegment, model: whisper.Whisper, lang: str) -> None:
    segment.audio.export("temp")
    segment.content = model.transcribe("temp", language=lang, fp16=False if torch.get_default_device() == "cpu" else True)


def transcribe_all_segments(timestamps: list[DiarizedSegment], model: whisper.Whisper, lang: str) -> None:
    for segment in timestamps:
        transcribe_segment(segment, model, lang)
    Path("temp").unlink(True)


def main() -> None:
    args = cli()

    logging.basicConfig(format="%(message)s",level=args.loglevel)

    set_torch_device(args.torch_device)

    logging.info(f"Using {torch.get_default_device()} device")

    audio_wav = AudioSegment.from_file(Path(args.audio))

    logging.info("Getting timestamps (this may take a while)")

    timestamps = join_timestamps(parse_speakers(audio_wav, read_auth_file(args.auth_token) if args.auth_file else args.auth_token))

    torch.set_default_device(args.torch_device)

    model = whisper.load_model(args.model)

    transcribe_all_segments(timestamps, model, args.language)

    full_transcript = join_transcripts(timestamps)

    logging.info("Saving transcript")

    save_to_file(full_transcript, Path(args.output_file))


if __name__ == "__main__":
    try:
        with warnings.catch_warnings(action="ignore"):
            main()
            exit(0)
    except Exception as e:
        logging.critical(e)
        exit(1)