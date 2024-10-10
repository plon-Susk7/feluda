LANGUAGES = {
    "en": "english",
    "hi": "hindi",
    "ta": "tamil",
    "te": "telugu",
}

def extract_audio_from_video(video_path):

    import ffmpeg
    
    (
        ffmpeg.input(video_path)
        .output("./sample_data/temp_audio.wav",format='wav')
        .run(quiet=True,overwrite_output=True)
    )

    return 

def extract_speech(fname):
    """Detect and export voice activity from an audio file.

    Args:
        fname (str): Path to audio file.

    Returns:
        str or bool: Name of the audio file with the extracted speech, False if no voice activity detected.
    """
    # get speech timestamps using our VAD model...
    get_speech_timestamps, _, read_audio, *_ = utils
    audio = read_audio(fname, sampling_rate=16000)
    speech_timestamps = get_speech_timestamps(
        audio, vad, sampling_rate=16000, return_seconds=True
    )

    # return false if no speech detected:
    if not speech_timestamps:
        return False

    # merge timestamps that are closer than a second for leniency...
    merged_timestamps = []
    current_segment = speech_timestamps[0]
    for next_segment in speech_timestamps[1:]:
        if next_segment['start'] - current_segment['end'] <= 1:
            current_segment['end'] = next_segment['end']
        else:
            merged_timestamps.append(current_segment)
            current_segment = next_segment
    merged_timestamps.append(current_segment)

    # isolate the speech as audio...
    with open(fname, 'rb') as file:
        global audio_segment
        audio_segment = AudioSegment.from_file(file, format="wav")
    segments = []
    duration = 0
    for ts in merged_timestamps:
        start = ts["start"] * 1000
        end = ts["end"] * 1000
        segment = audio_segment[start:end]
        segments.append(audio_segment[start:end])
        duration += len(segment)
        if duration > 30000:
            # exit the loop if we have an audio atleast 30 seconds long
            break
    final_audio = sum(segments, AudioSegment.empty())

    # Export audio as a tmp file...
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as speech:
        final_audio.export(speech.name, format="wav")
    return speech.name

def detect_language(fname):
    """Detect language of from an audio file using whisper.

    Returns:
        str: Detected ISO 639-1 language code.
    """
    # load and normalize audio to fit 30 seconds duration
    audio = whisper.load_audio(fname)
    audio = whisper.pad_or_trim(audio)

    # create log-Mel spectrogram
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect language
    _, probs = model.detect_language(mel)
    return max(probs, key=probs.get)

def initialize(param):
    global whisper, model, AudioSegment, utils, vad, os, tempfile

    import os
    import tempfile
    import whisper
    import torch
    from pydub import AudioSegment

    model = whisper.load_model("base")
    vad, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")

def run(audio_file):
    extract_audio_from_video(audio_file["path"])
    print(audio_file["path"])
    audio = audio_file["path"]
    speech = extract_speech(audio)
    if speech:
        # audio contains voice activity
        try:
            language_id = detect_language(speech)
            language = LANGUAGES[language_id] # get the generic name from id
            return {"id": language_id, "language": language}
        finally:
            os.remove(speech)
    return {"id": "und", "language": "undefined"}



