"""
Extract spatio-semantic features from Cookie Theft picture description transcripts.

Extends the ICASSP 2023 pipeline with refined keyword matching and an automatic
punctuation restoration step for RTF transcripts.

Differences from the ICASSP pipeline
--------------------------------------
- Sentence segmentation for RTF files uses a deep-learning punctuation restoration
  model (deepmultilingualpunctuation) rather than treating the whole document as
  one block. CHAT .cha files use *PAR: utterance boundaries as before.
- Additional post-hoc keyword corrections applied to every sentence:
    - 'cookie jar': counts jar (unit 7) only, not cookie (unit 6) separately.
    - 'fall': unit 18 (boy/stool falling) retained only when preceded by boy or girl.
    - 'reach': reassigned from girl-gesture (21) to boy-taking (17) when boy is present.
    - Units 22/23: retained only when required co-occurring units are present.
- Keyword list refined to reduce cross-unit overlap (e.g., 'plate'/'dish'/'floor').

Supported transcript formats
-----------------------------
- CHAT (.cha): standard DementiaBank format; *PAR: utterances are parsed
  sentence-by-sentence.
- Plain-text RTF (.rtf): punctuation model segments speech into sentences.

Usage
-----
1. Point `datasets` at your transcript directory.
2. Point `results_dir` at a (writable) output directory.
3. Run:  python extract_feat_0509.py

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
from utils import *


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
all_df.to_csv('./spatio_semantic_features.csv', index=False)

plot_spatio_semantics(dir=results_dir)
