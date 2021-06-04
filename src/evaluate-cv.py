#%%
import torch
import torchaudio
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import re

dataset = load_dataset("common_voice", "mn", split="test")
wer = load_metric("wer")

#%%
processor = Wav2Vec2Processor.from_pretrained("wav2vec2-large-xlsr-53-mongolian-trainer")
model = Wav2Vec2ForCTC.from_pretrained("wav2vec2-large-xlsr-53-mongolian-trainer")
model.to("cuda")

#%%
chars_to_ignore_regex = r'[\,\?\.\!\-\;\:\"\»\'\«]'
resampler = torchaudio.transforms.Resample(48_000, 16_000)


# Preprocessing the datasets.
# We need to read the aduio files as arrays
def speech_file_to_array_fn(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resampler(speech_array).squeeze().numpy()
    return batch


dataset = dataset.map(speech_file_to_array_fn)

#%%


# Preprocessing the datasets.
# We need to read the aduio files as arrays
def evaluate(batch):
    inputs = processor(
        batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True
    )

    with torch.no_grad():
        logits = model(
            inputs.input_values.to("cuda"),
            attention_mask=inputs.attention_mask.to("cuda"),
        ).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_strings"] = processor.batch_decode(pred_ids)
    return batch


results = dataset.map(evaluate, batched=True, batch_size=8)

#%%
wer_metric_result = wer.compute(
    predictions=results["pred_strings"], references=results["sentence"]
)
print(f"Test WER: {wer_metric_result:.2%}")

# %%
sample_result = evaluate(dataset[12])
wer_metric_result = wer.compute(
    predictions=[sample_result["pred_strings"][0]],
    references=[sample_result["sentence"]],
)
print(f"Sample WER: {wer_metric_result:.2%}")
print(f"Predicted : {sample_result['pred_strings'][0]}")
print(f"Target    : {sample_result['sentence']}")

# %%
