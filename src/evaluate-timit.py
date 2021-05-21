#%%
import torch
import torchaudio
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import re


# Load data
timit = load_dataset("timit_asr", split='test').remove_columns(
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
processor = Wav2Vec2Processor.from_pretrained("./wav2vec2-base-timit-demo-trainer")
model = Wav2Vec2ForCTC.from_pretrained("./wav2vec2-base-timit-demo-trainer")
model.to("cuda")
wer = load_metric("wer")

chars_to_ignore_regex = r'[\,\?\.\!\-\;\:\"]'

#%%


# Preprocessing the datasets.
def speech_file_to_array_fn(batch):
    """We need to read the audio files as arrays"""
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower()
    speech_array, sampling_rate = torchaudio.load(batch["file"])
    batch["speech"] = speech_array
    batch["sampling_rate"] = sampling_rate
    return batch


timit = timit.map(speech_file_to_array_fn, num_proc=4)

#%%


def evaluate(batch):
    model.to("cuda")
    input_values = processor(
        batch["speech"], sampling_rate=batch["sampling_rate"], return_tensors="pt"
    ).input_values.to("cuda")

    with torch.no_grad():
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_text"] = processor.batch_decode(pred_ids)[0]

    return batch


results = timit.map(evaluate)

#%%
wer_metric_result = wer.compute(
    predictions=results["pred_text"], references=results["text"]
)
print(f"Test WER: {wer_metric_result:.2%}")

# %%

sample_result = evaluate(timit[21])
wer_metric_result = wer.compute(
    predictions=[sample_result["pred_text"]], references=[sample_result["text"]]
)
print(f"Sample WER: {wer_metric_result:.2%}")
print(f"Predicted : {sample_result['pred_text']}")
print(f"Target    : {sample_result['text']}")
