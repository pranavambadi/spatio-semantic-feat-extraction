"""
Extract spatio-semantic features from Cookie Theft picture description transcripts.

Implements the feature extraction pipeline from:
  Ambadi et al., "Spatio-Semantic Analysis of Picture Description for Dementia
  Detection", ICASSP 2023.

Each transcript is mapped to 23 semantic units (subjects, places, objects, actions)
from the Cookie Theft picture via keyword matching. Sequential transitions between
units are modeled as a directed graph, from which path distance, node coverage,
cycle, and quadrant-crossing features are computed.

Supported transcript formats
-----------------------------
- CHAT (.cha): standard DementiaBank format; *PAR: utterances are parsed
  sentence-by-sentence.
- Plain-text RTF (.rtf): entire document is treated as participant speech.

Usage
-----
1. Point `datasets` at your transcript directory.
2. Point `results_dir` at a (writable) directory for intermediate .pkl files.
3. Run:  python extract_feat_ICASSP.py

Output
------
A CSV with one row per transcript containing CIU_seq, all graph-theoretic
spatio-semantic features, and participant metadata (mmse, sex, age) where
available from .cha files.
"""

import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from utilsICASSP import *


# datasets = './Nathalie_cookie_theft'   # RTF transcripts
datasets = './WRAP'                      # CHAT .cha transcripts
results_dir = './results'
os.makedirs(results_dir, exist_ok=True)

files = [f'{datasets}/{f}' for f in os.listdir(datasets)]
records = [extract_data(f) for f in files]
all_df = pd.DataFrame(records)

all_df = extract_spatio_semantics(all_df, results_dir)
spatial_semantics = summarize_spatio_semantics(dir=results_dir)
all_df = all_df.merge(spatial_semantics, on='file_id')
all_df = all_df.drop(columns='file_id')
all_df.to_csv('./WRAP_spatio_semantic_features.csv', index=False)

plot_spatio_semantics(dir=results_dir)
