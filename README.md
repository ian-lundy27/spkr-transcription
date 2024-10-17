## Overview

This script uses speaker diarization and the OpenAI Whisper model to create a text transcript broken down by speaker based on an input audio file.

## Installation

Usage requires a free api token from [huggingface](https://huggingface.co/) and access to the [speaker diarization](https://huggingface.co/pyannote/speaker-diarization-3.1) and [segmentation](https://huggingface.co/pyannote/segmentation-3.0) models.

This project depends on several python packages and [ffmpeg](https://www.ffmpeg.org/).

Creating a [virtual environment](https://docs.python.org/3/library/venv.html#creating-virtual-environments) is recommended as the dependencies take up >1.5 GB, excluding cached whisper models.

If using a virtual environment, be sure to [activate](https://docs.python.org/3/library/venv.html#how-venvs-work) the environment in your terminal before installing python dependencies and running the script.

```pip install openai-whisper pyannote.audio pydub```

[ffmpeg](https://www.ffmpeg.org/) can be installed from the website or through a package manager of your choice.

```choco install ffmpeg```

```brew install ffmpeg```

## Usage

Your cli call will follow the format ```python transcribe.py <path/to/audio/file> <auth token> <args>```

If storing your api token in a file, use the file path as the token argument and include the flag --auth-file.

Using cuda or the cpu is recommended. Other apis such as mps may not be supported. 

Example usage:
```python transcribe.py audio.wav auth.txt --auth-file --model=turbo```

Run ```python transcribe.py -h``` for more information

The resulting transcript will save to a file, ```output.txt``` by default.

## Is this stupid?

Yes.
