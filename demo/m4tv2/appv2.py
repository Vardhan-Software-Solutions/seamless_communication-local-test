#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from __future__ import annotations
import sys
sys.path.append('../../src')
# sys.path.append("src/seamless_communication")

import os
import pathlib
import getpass

import gradio as gr
import numpy as np
import torch
import torchaudio
from fairseq2.assets import InProcAssetMetadataProvider, asset_store
from huggingface_hub import snapshot_download
from seamless_communication.inference import Translator

from lang_list import (
    ASR_TARGET_LANGUAGE_NAMES,
    LANGUAGE_NAME_TO_CODE,
    S2ST_TARGET_LANGUAGE_NAMES,
    S2TT_TARGET_LANGUAGE_NAMES,
    T2ST_TARGET_LANGUAGE_NAMES,
    T2TT_TARGET_LANGUAGE_NAMES,
    TEXT_SOURCE_LANGUAGE_NAMES,
)

user = getpass.getuser() # this is not portable on windows
CHECKPOINTS_PATH = pathlib.Path(os.getenv("CHECKPOINTS_PATH", f"/home/{user}/app/models"))
if not CHECKPOINTS_PATH.exists():
    snapshot_download(repo_id="facebook/seamless-m4t-v2-large", repo_type="model", local_dir=CHECKPOINTS_PATH)
asset_store.env_resolvers.clear()
asset_store.env_resolvers.append(lambda: "demo")
demo_metadata = [
    {
        "name": "seamlessM4T_v2_large@demo",
        "checkpoint": f"file://{CHECKPOINTS_PATH}/seamlessM4T_v2_large.pt",
        "char_tokenizer": f"file://{CHECKPOINTS_PATH}/spm_char_lang38_tc.model",
    },
    {
        "name": "vocoder_v2@demo",
        "checkpoint": f"file://{CHECKPOINTS_PATH}/vocoder_v2.pt",
    },
]
asset_store.metadata_providers.append(InProcAssetMetadataProvider(demo_metadata))

DESCRIPTION = """\
# SeamlessM4T
[SeamlessM4T](https://github.com/facebookresearch/seamless_communication) is designed to provide high-quality
translation, allowing people from different linguistic communities to communicate effortlessly through speech and text.
This unified model enables multiple tasks like Speech-to-Speech (S2ST), Speech-to-Text (S2TT), Text-to-Speech (T2ST)
translation and more, without relying on multiple separate models.
"""

CACHE_EXAMPLES = os.getenv("CACHE_EXAMPLES") == "1" and torch.cuda.is_available()

AUDIO_SAMPLE_RATE = 16000.0
MAX_INPUT_AUDIO_LENGTH = 60  # in seconds
DEFAULT_TARGET_LANGUAGE = "French"

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32

translator = Translator(
    model_name_or_card="seamlessM4T_v2_large",
    vocoder_name_or_card="vocoder_v2",
    device=device,
    dtype=dtype,
    apply_mintox=True,
)


def preprocess_audio(input_audio: str) -> None:
    arr, org_sr = torchaudio.load(input_audio)
    new_arr = torchaudio.functional.resample(arr, orig_freq=org_sr, new_freq=AUDIO_SAMPLE_RATE)
    max_length = int(MAX_INPUT_AUDIO_LENGTH * AUDIO_SAMPLE_RATE)
    if new_arr.shape[1] > max_length:
        new_arr = new_arr[:, :max_length]
        gr.Warning(f"Input audio is too long. Only the first {MAX_INPUT_AUDIO_LENGTH} seconds is used.")
    torchaudio.save(input_audio, new_arr, sample_rate=int(AUDIO_SAMPLE_RATE))


OUTPUT_FILE = "s2tt_output.txt"

def run_s2tt(input_audio: str, source_language: str, target_language: str) -> str:
    preprocess_audio(input_audio)
    source_language_code = LANGUAGE_NAME_TO_CODE[source_language]
    target_language_code = LANGUAGE_NAME_TO_CODE[target_language]
    out_texts, _ = translator.predict(
        input=input_audio,
        task_str="S2TT",
        src_lang=source_language_code,
        tgt_lang=target_language_code,
    )

    print(" OUTPUT --> ",out_texts[0])
     # Write the output to a file
    try:
        with open(OUTPUT_FILE, 'a') as f:  # 'a' mode appends to the file
            f.write(f"Input Audio: {input_audio}\n")
            f.write(f"Source Language: {source_language} ({source_language_code})\n")
            f.write(f"Target Language: {target_language} ({target_language_code})\n")
            f.write(f"Translation: {out_texts[0]}\n")
            f.write("=" * 40 + "\n")  # Add a separator between entries

    except Exception as e:
        print(f"Error writing to {OUTPUT_FILE}: {e}")

    return str(out_texts[0])


run_s2tt("2079-11-21_3.wav","Nepali","English")
