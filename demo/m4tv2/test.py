from __future__ import annotations
import sys
sys.path.append('../../src')

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