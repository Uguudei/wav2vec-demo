"""Train model"""
#%%
import json
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

# import IPython.display as ipd
import numpy as np
import pandas as pd
import torch
import torchaudio
from datasets import load_dataset, load_metric
from IPython.display import HTML, display
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

# from src.vscode_audio import ipython_audio

#%%


def prepare_common_voice(split):
    """download and load common voice dataset from hugging face"""
    dataset = load_dataset("common_voice", "mn", split=split)
    dataset = dataset.remove_columns(
        [
            'accent',
            'age',
            'client_id',
            'down_votes',
            'gender',
            'locale',
            'segment',
            'up_votes',
        ]
    )
    dataset = dataset.rename_column('path', 'file')
    dataset = dataset.rename_column('sentence', 'text')
    return dataset


dataset_train = prepare_common_voice('train+validation')
dataset_test = prepare_common_voice('test')
print('Train:', dataset_train)
print('Test:', dataset_test)

#%%


def show_random_elements(dataset, num_examples=10):
    """show random n rows from dataset"""
    assert num_examples <= len(
        dataset
    ), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    data = pd.DataFrame(dataset[picks])
    display(HTML(data.to_html()))


show_random_elements(dataset_train.remove_columns(["file"]))

#%%

CHARS_TO_IGNORE_REGEX = r'[\,\?\.\!\-\;\:\"\»\'\«]'


def remove_special_characters(batch):
    """replace special characters with none"""
    batch["text"] = re.sub(CHARS_TO_IGNORE_REGEX, '', batch["text"]).lower()
    return batch


dataset_train = dataset_train.map(remove_special_characters)
dataset_test = dataset_test.map(remove_special_characters)
show_random_elements(dataset_train.remove_columns(["file"]))


def replace_characters(batch):
    """replace latin 'h' with 'х' cyrillic"""
    batch["text"] = batch["text"].replace('h', 'х')
    return batch


dataset_train = dataset_train.map(replace_characters)

#%%


def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


vocab_train = dataset_train.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=dataset_train.column_names,
)
vocab_test = dataset_test.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=dataset_test.column_names,
)

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(vocab_list)}
print(vocab_dict)

#%%
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
print(len(vocab_dict))

#%%
with open('./data/vocab_mn.json', 'w', encoding='UTF-8') as vocab_file:
    json.dump(vocab_dict, vocab_file)

#%%
tokenizer = Wav2Vec2CTCTokenizer(
    "data/vocab_mn.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
)

#%%
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

#%% [markdown]
### Preprocess Data

#%%
print(dataset_train[0])

#%%
resampler = torchaudio.transforms.Resample(48_000, 16_000)


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["file"])
    batch["speech"] = resampler(speech_array).squeeze().numpy()
    batch["sampling_rate"] = 16_000
    batch["target_text"] = batch["text"]
    return batch


dataset_train = dataset_train.select(list(range(0, 100))).map(
    speech_file_to_array_fn,
    remove_columns=dataset_train.column_names,
    num_proc=16,
)
dataset_test = dataset_test.map(
    speech_file_to_array_fn,
    remove_columns=dataset_test.column_names,
    num_proc=16,
)

#%%

rand_int = random.randint(0, len(dataset_train))
sample_audio = np.asarray(dataset_train[rand_int]["speech"])

# display(ipd.Audio(data=sample_audio, autoplay=True, rate=16000))
# ipython_audio(sample_audio, 16000)

#%%
rand_int = random.randint(0, len(dataset_train))

print("Target text:", dataset_train[rand_int]["target_text"])
print("Input array shape:", np.asarray(dataset_train[rand_int]["speech"]).shape)
print("Sampling rate:", dataset_train[rand_int]["sampling_rate"])

#%%


def prepare_dataset(batch):
    # check that all files have the correct sampling rate
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(
        batch["speech"], sampling_rate=batch["sampling_rate"][0]
    ).input_values

    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch


dataset_train = dataset_train.map(
    prepare_dataset,
    remove_columns=dataset_train.column_names,
    batch_size=8,
    num_proc=4,
    batched=True,
)
dataset_test = dataset_test.map(
    prepare_dataset,
    remove_columns=dataset_test.column_names,
    batch_size=8,
    num_proc=4,
    batched=True,
)

#%%


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side
            and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch


#%%

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

#%%

cer_metric = load_metric("cer")

#%%


def compute_metrics(pred):
    """compute score metrics"""
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}


#%%
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53",
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    gradient_checkpointing=True,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)

#%%
model.freeze_feature_extractor()

#%%
training_args = TrainingArguments(
    output_dir="./wav2vec2-large-xlsr-53-mongolian",
    group_by_length=True,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=100,
    fp16=True,
    save_steps=500,
    eval_steps=500,
    logging_steps=500,
    learning_rate=1e-4,
    # weight_decay=0.005,
    warmup_steps=1000,
    save_total_limit=2,
)

#%%
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    tokenizer=processor.feature_extractor,
)

#%%
trainer.train()

# Save model to load later
trainer.save_model(output_dir="./wav2vec2-large-xlsr-53-mongolian")
processor.save_pretrained(("./wav2vec2-large-xlsr-53-mongolian"))
