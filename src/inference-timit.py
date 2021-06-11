import torch
import torchaudio
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load data
dataset = load_dataset("timit_asr", split='test').remove_columns(
    [
        "phonetic_detail",
        "word_detail",
        "dialect_region",
        "id",
        "sentence_type",
        "speaker_id",
    ]
)

# Load model
processor = Wav2Vec2Processor.from_pretrained("./wav2vec2-base-timit")
model = Wav2Vec2ForCTC.from_pretrained("./wav2vec2-base-timit")


# Preprocessing the datasets.
def speech_file_to_array_fn(batch):
    """We need to read the audio files as arrays"""
    batch["speech"], batch["sampling_rate"] = torchaudio.load(batch["file"])
    return batch


dataset = dataset.map(speech_file_to_array_fn)
inputs = processor(
    dataset["speech"][0], sampling_rate=16_000, return_tensors="pt", padding=True
)

with torch.no_grad():
    logits = model(inputs.input_values).logits

predicted_ids = torch.argmax(logits, dim=-1)

print("Prediction:", processor.batch_decode(predicted_ids))
print("Reference:", dataset["text"][0])
