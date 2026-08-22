## citations: deepmultilingualpunctuation

import regex as re
import warnings, os, string
from itertools import groupby
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance
from statistics import mean
import math
import spacy
warnings.filterwarnings('ignore')

# ── Cookie Theft picture description task — 23 semantic units ─────────────
# Covers subjects (1-3), places (4-5), objects (6-16), and actions (17-23).
nodes = unit = [
    'boy', 'girl', 'woman', 'kitchen', 'outside', 'cookie', 'jar', 'stool',
    'sink', 'plate', 'dishcloth', 'water', 'window', 'cupboard', 'dishes',
    'curtains', 'boy taking/stealing', 'boy or stool falling',
    'woman drying/washing plates', 'water overflowing',
    'action performed by girl', 'woman unconcerned by overflowing',
    'woman indifferent to the children'
]

# Maps each unit name to its numeric label string used in visualizations
mapping = {
    'boy': "1", 'girl': "2", 'woman': "3", 'kitchen': "4", 'outside': "5",
    'cookie': "6", 'jar': "7", 'stool': "8", 'sink': "9", 'plate': "10",
    'dishcloth': "11", 'water': "12", 'window': "13", 'cupboard': "14",
    'dishes': "15", 'curtains': "16", 'boy taking/stealing': "17",
    'boy or stool falling': "18", 'woman drying/washing plates': "19",
    'water overflowing': "20", 'action performed by girl': "21",
    'woman unconcerned by overflowing': "22,23",
    'woman indifferent to the children': " 22,23 "
}

# (x, y) pixel coordinates of each unit's canonical position in the Cookie Theft image
lookup = [
    (183, 118), (82, 220), (324, 171), (387, 193), (440, 160), (123, 41),
    (123, 59), (145, 268), (387, 216), (354, 143), (352, 171), (378, 274),
    (430, 119), (134, 75), (488, 243), (433, 78), (145, 64), (165, 226),
    (340, 156), (384, 233), (85, 181), (331, 76), (331, 76)
]

nlp = spacy.load('en_core_web_sm')

# Keyword lists for each of the 23 semantic units (index i → unit i+1).
# Words are matched against lemmatized transcript tokens.
# Note: "plate" removed from unit 15 (dishes) and "dish"/"floor" removed from
# units 10/12 to avoid cross-unit overlap.
keyword_list = [
    ["boy", "his", "he", "brother", "son", "child", "shirt", "shoe", "sock",
     "kid", "man", "himself", "adolescent", "male", "guy", "him", "child"],
    ["girl", "sister", "daughter", "skirt", "dress", "child", "kid"],
    ["mother", "mom", "woman", "lady", "ladys", "mama", "wife", "adult",
     "domestic", "homemaker", "women", "housewife", "moms", "dress", "apron",
     "fifties", "nineteen-fifties", "heels", "high-heels"],
    ["culinary", "interior", "kitchen", "home", "counter"],
    ["outside", "tree", "grass", "bush", "yard", "summer", "grow", "path",
     "pathway", "sidewalk", "shrub", "backyard", "lawn", "shrubbery", "snow",
     "spring", "garden", "walkway", "garage", "flower", "background", "sunny",
     "leave", "exterior", "scenery", "day", "outdoor"],
    ["cookie", "pastry", "snack", "cookies", "chocolatechip"],
    ["canister", "container", "holder", "jar", "vessel", "lid"],
    ["stool", "stepstool", "threelegged", "three", "chair", "footstool",
     "bench", "furniture", "ladder", "perch", "step ladder", "step stool",
     "stepladder", "step-stool", "step-ladder"],
    ["basin", "drain", "sink", "sinks", "faucet"],
    ["plate", "saucer"],                                    # "dish" removed — overlaps unit 15
    ["cloth", "dishcloth", "napkin", "rag", "textile", "towel", "dishtowel",
     "handtowel", "dishrag", "sponge"],
    ["deluge", "flood", "flow", "inundation", "liquid", "moisture", "torrent",
     "water", "waters", "puddle"],                         # "floor" removed — spurious matches
    ["casement", "glass", "pane", "window"],
    ["cabinet", "cupboard", "shelf", "storage", "door", "handle"],
    ["dish", "cup", "bowl"],                               # "plate" removed — overlaps unit 10
    ["curtain", "drape", "fabric", "textile", "window dressing", "wind",
     "hang", "wave", "tie", "tieback"],
    ["acquire", "snatch", "extract", "grab", "secure", "steal", "take",
     "taking", "climb", "raid", "sneak", "snitch", "rob", "try"],
    ["collapse", "collapsing", "fall", "tilt", "tip", "tipping", "topple",
     "toppling", "unstable", "overturn", "hurt", "backwards", "balance", "crash"],
    ["clean", "dry", "rinse", "scrub", "wash", "wipe", "washing"],
    ["facuet,", "deluge", "flood", "flow", "inundation", "overflow", "spill",
     "spillage", "splash", "torrent", "overflowing", "run", "pour", "overrun",
     "spilling", "drip", "overfill"],
    ["shh", "gesture", "motion", "reach", "signal", "laugh", "speak",
     "shout", "warn", "request", "ask", "finger", "mouth", "lips", "quiet",
     "say", "tell", "lip", "giggle", "point"],
    ["daze", "disregard", "ignorant", "neglectful", "nonchalant", "oblivious",
     "unconcerned", "notice", "aware", "unaware", "attention", "realize",
     "concerned", "daydream", "distracted", "care", "ignore", "concern", "distract"],
    ["doesn't see", "apathetic", "attention", "distracted", "disregard",
     "focus", "ignorant", "indifferent", "nonchalant", "oblivious",
     "unconcerned", "aware", "unaware", "notice", "unconcern", "distract",
     "ignore", "turn", "clue"]          # "back"/"behind" removed — too generic
]


def remove_punctuation_translate(input_string):
    """Remove all punctuation characters from a string."""
    remove_punct_map = dict.fromkeys(map(ord, string.punctuation))
    return input_string.translate(remove_punct_map)


def expand_repeats(input_text):
    """Expand CHAT-format repeat markers, e.g. 'word [x 3]' → 'word word word'."""
    words = input_text.split()
    result = ""
    for word in words:
        if "[x" in word:
            repeat_count = int(words[words.index("[x") + 1][0]) - 1
            base_word = words[words.index("[x") - 1] + ' '
            result += base_word * repeat_count + " "
        else:
            result += word + " "
    return result.strip()


def _clean(s):
    """Strip CHAT-format annotations, time codes, ellipses, and non-lexical tokens."""
    s = s.replace('.', ' ')
    s = re.sub('\x15\d*_\d*\x15', ' ', s)
    s = re.sub('\[.*?\]', ' ', s)
    s = s.strip()
    s = re.sub('\t|\n|<|>', ' ', s)
    s = re.sub('\d+', '&=', s)
    s = re.sub('‡', '&=', s)
    s = re.sub('\&', '&=', s)
    s = re.sub('xxx', '&=', s)
    tmp = re.sub('\&\=.*?\ ', ' ', s)
    tmp = re.sub(r'…', ' ', tmp)
    tmp = remove_punctuation_translate(tmp)
    return tmp


def extract_vec(text):
    """Fit a TF-IDF vectorizer on a list of strings and return the feature matrix."""
    vectorizer = TfidfVectorizer(max_features=5)
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()


def hasNumbers(inputString):
    """Return True if the string contains any digit character."""
    return any(char.isdigit() for char in inputString)


def remove_adjacent_duplicates(lst):
    """Remove consecutive duplicate elements from a list."""
    return [key for key, group in groupby(lst)]


def match_keyword_and_count(transcript):
    """Return a list of unit indices (1-based) for every keyword match in the transcript.

    Iterates over each word in the (space-split) transcript and checks it against
    all 23 keyword lists. The same unit index may appear multiple times if its
    keywords occur multiple times.
    """
    global lookup, unit, nodes, keyword_list, mapping
    count = []
    for word in transcript.split(' '):
        for x in range(len(keyword_list)):
            if word in keyword_list[x]:
                count.append(x + 1)
    return count


def process_ciu_reach(lst):
    """Remap a unit-21 match to unit 17 when the boy (unit 1) preceded it.

    When 'reach' is present and unit 1 (boy) appeared before unit 21 (girl
    gesture) in the same sentence, the sequence is rewritten so that unit 21
    becomes unit 17 (boy taking/stealing) and stray unit-2 (girl) entries after
    the last unit-1 are dropped.
    """
    result = []
    last_1_index = None
    for num in lst:
        if num == 1:
            result.append(1)
            last_1_index = len(result) - 1
        elif num == 21:
            if last_1_index is not None:
                result = result[:last_1_index + 1] + [x for x in result[last_1_index + 1:] if x != 2]
                result.append(17)
            else:
                result.append(21)
        else:
            result.append(num)
    return result


def extract_data(file_name, time=False):
    """Parse a transcript file and return a dict of speech and metadata fields.

    Supports CHAT-format .cha files and plain-text .rtf files.

    For .cha files, extracts per-utterance participant speech and participant
    metadata (mmse, sex, age) from the @ID header. For .rtf files, converts
    RTF markup to plain text and treats the whole document as participant speech.

    Parameters
    ----------
    file_name : str
        Path to the transcript file (.cha or .rtf).
    time : bool
        If True, also extract per-sentence timing fields (CHAT format only).

    Returns
    -------
    dict
        Keys include file_id, file_dir, joined_all_par_speech (lemmatized),
        joined_all_par_speech_ori (raw), and for .cha files: mmse, sex, age.
    """
    from striprtf.striprtf import rtf_to_text
    global nlp

    par = {}
    file_id = re.split(r'_|\.', os.path.basename(file_name))[0].lower()
    par['file_id'] = file_id
    par['file_dir'] = file_name

    if file_name.endswith('.rtf'):
        with open(file_name, 'r', errors='ignore') as f:
            plain_text = rtf_to_text(f.read())
        s = _clean(expand_repeats(plain_text))
        par['speech'] = par['clean_speech'] = par['clean_par_speech'] = []
        par['joined_all_speech'] = plain_text
        par['joined_all_par_speech_ori'] = plain_text
        par['joined_all_par_speech'] = ' '.join([token.lemma_ for token in nlp(s)])
        return par

    f = iter(open(file_name, encoding='utf-8', errors='ignore'))
    l = next(f)
    speech = []

    try:
        curr_speech = ''
        while True:
            if l.startswith('@ID'):
                participant = [i.strip() for i in l.split('|')]
                if participant[2] == 'PAR':
                    par['mmse'] = float(participant[8]) if participant[8] else ''
                    par['sex'] = participant[4][0] if participant[4] else ''
                    par['age'] = int(participant[3][0:2]) if participant[3] else ''
            if l.startswith('*PAR:'):
                curr_speech += l
            elif len(curr_speech) != 0 and not (l.startswith('%') or l.startswith('*') or l.startswith('@')):
                curr_speech += l
            elif len(curr_speech) != 0 and (l.startswith('%') or l.startswith('*') or l.startswith('@')):
                speech.append(curr_speech)
                curr_speech = ''
            if ' '.join(str(l).split()) == '@G: Cat':
                break
            l = next(f)
    except StopIteration:
        pass

    clean_par_speech, clean_all_speech = [], []
    par_speech_time_segments = []
    is_par = False
    for s in speech:
        def _parse_time(s):
            try:
                tmp = [*map(int, re.search('\x15(\d*_\d*)\x15', s).groups()[0].split('_'))]
                return tmp
            except:
                return ''

        if s.startswith('*PAR:'):
            is_par = True
        elif s.startswith('*INV:'):
            is_par = False
            s = re.sub('\*INV:\t', ' ', s)
        if is_par:
            s = re.sub('\*PAR:\t', ' ', s)
            s = expand_repeats(s)
            if time:
                tmp = _parse_time(s)
                if tmp != '':
                    par_speech_time_segments.append(tmp)
                    clean_par_speech.append(_clean(s))
            else:
                clean_par_speech.append(_clean(s))
        clean_all_speech.append(_clean(s))

    par['speech'] = speech
    par['clean_speech'] = clean_all_speech
    par['clean_par_speech'] = clean_par_speech
    par['joined_all_speech'] = ' '.join(clean_all_speech)
    par['joined_all_par_speech_ori'] = ' '.join(clean_par_speech)

    if time:
        par['per_sent_times'] = [
            par_speech_time_segments[i][1] - par_speech_time_segments[i][0]
            for i in range(len(par_speech_time_segments))
        ]
        par['total_time'] = par_speech_time_segments[-1][1] - par_speech_time_segments[0][0]
        par['time_before_par_speech'] = par_speech_time_segments[0][0]
        par['time_between_sents'] = [
            0 if i == 0 else max(0, par_speech_time_segments[i][0] - par_speech_time_segments[i - 1][1])
            for i in range(len(par_speech_time_segments))
        ]

    doc = nlp(par['joined_all_par_speech_ori'])
    par['joined_all_par_speech'] = ' '.join([token.lemma_ for token in doc])
    return par


def extract_spatio_semantics_asu(dir, txtfiles, results_dir='results'):
    """Extract spatio-semantic features from plain-text RTF transcripts.

    Uses a deep-learning punctuation restoration model to split each transcript
    into pseudo-sentences, then matches each sentence against the 23 semantic unit
    keyword lists. Applies post-hoc corrections for ambiguous keywords:
      - 'cookie jar': counts jar (unit 7) but not cookie (unit 6)
      - 'fall': unit 18 (boy/stool falling) only if immediately preceded by boy or girl
      - 'reach': remapped via process_ciu_reach when boy is present
      - units 22/23: only retained when required co-occurring units are present
          unit 22 requires one of: sink (9), water overflowing (20), water (12)
          unit 23 requires one of: boy (1), girl (2), cookie (6), stool (8), falling (18)

    Saves a per-file pickle DataFrame to `results_dir/dataframes/`.
    Returns a DataFrame with CIU_seq (list of unit-name sequences) and text columns.

    Parameters
    ----------
    dir : str
        Directory containing the RTF transcript files.
    txtfiles : list of str
        Filenames within `dir` to process.
    results_dir : str
        Root directory under which `dataframes/` will be created for output pickles.
    """
    from deepmultilingualpunctuation import PunctuationModel
    from striprtf.striprtf import rtf_to_text
    model = PunctuationModel()
    global lookup, unit, nodes, keyword_list, mapping

    dataframes_dir = os.path.join(results_dir, 'dataframes')
    os.makedirs(dataframes_dir, exist_ok=True)

    all_df = pd.DataFrame(txtfiles)
    entirelist = []
    alltxt = []
    pbar = tqdm(total=len(txtfiles))

    # Required co-occurring unit indices for units 22 and 23
    required_22 = {9, 20, 12}
    required_23 = {1, 2, 6, 8, 18}

    for txt in txtfiles:
        with open(os.path.join(dir, txt), 'r', errors='ignore') as f:
            curr_speech = rtf_to_text(f.read())
        curr_speech = curr_speech.lower()
        if not curr_speech.endswith('.'):
            curr_speech += '.'
        alltxt.append(curr_speech)

        mylist = []
        entiretextlist = []

        s = expand_repeats(curr_speech)
        s = _clean(s)
        s = re.sub(r'\s+', ' ', s).strip()

        result = model.restore_punctuation(s)
        for s in re.split(r'[.,]', result):
            doc = nlp(s)
            s = ' '.join([token.lemma_ for token in doc])

            try:
                templist = match_keyword_and_count(s)

                # 'cookie jar' → count jar only, not cookie separately
                if 'cookie jar' in s:
                    templist = [x for i, x in enumerate(templist)
                                if not (x == 6 and i + 1 < len(templist) and templist[i + 1] == 7)]

                # 'fall' triggers unit 18 only when boy or girl preceded it
                if 'fall' in s:
                    templist = [
                        x for i, x in enumerate(templist)
                        if x != 18 or (i > 0 and templist[i - 1] in {1, 2})
                    ]

                if 'reach' in s and len(templist) > 1:
                    templist = process_ciu_reach(templist)

                templist = [i for n, i in enumerate(templist) if i not in templist[:n]]

                if 22 in templist or 23 in templist:
                    has_required_22 = any(x in required_22 for x in templist)
                    has_required_23 = any(x in required_23 for x in templist)
                    templist = [x for x in templist
                                if (x != 22 or has_required_22) and (x != 23 or has_required_23)]

                temptxtlist = [unit[i - 1] for i in templist]
                entiretextlist.extend(temptxtlist)
                mylist.extend(templist)
            except:
                print(f"Error processing utterance in {txt}")

        new_list, unit_list, unit_labels, reference, quadrant = [], [], [], [], []
        entirelist.append(entiretextlist)

        if not mylist:
            new_list, unit_list, unit_labels, reference, quadrant = [(0, 0)], 'null', 'null', 'null', 'null'
        else:
            for x in mylist:
                try:
                    templook = lookup[x - 1]
                    new_list.append(templook)
                    unit_list.append(unit[x - 1])
                    unit_labels.append(x)
                    reference.append(
                        'subject' if x <= 3 else 'place' if 3 < x <= 5 else 'object' if 5 < x <= 16 else 'action')
                    quadrant.append(
                        1 if templook[0] < 273 and templook[1] < 195 else
                        2 if templook[0] < 273 and templook[1] > 195 else
                        3 if templook[0] > 273 and templook[1] < 195 else 4)
                except:
                    print("Fail to look up: {:d}".format(x))
                    pass

        df = pd.DataFrame(new_list, columns=['x', 'y'])
        df['order'] = np.arange(len(df)) + 1
        df['unit'], df['unit #'], df['reference'], df['quadrant'] = unit_list, unit_labels, reference, quadrant
        filename = os.path.splitext(txt)[0]
        df.to_pickle(os.path.join(dataframes_dir, filename + '.pkl'))
        pbar.update(1)

    all_df['CIU_seq'] = entirelist
    all_df['text'] = alltxt
    return all_df


def extract_spatio_semantics(all_df, dir):
    """Extract spatio-semantic features from transcripts via keyword matching.

    Supports both CHAT-format .cha files (parses *PAR: utterances as sentences)
    and plain-text .rtf files (uses a deep-learning punctuation restoration model
    to recover sentence boundaries). Applies the same post-hoc keyword corrections
    to both formats:
      - 'cookie jar': counts jar (unit 7) only, not cookie (unit 6) separately.
      - 'fall': unit 18 (boy/stool falling) only if preceded by boy or girl.
      - 'reach': remapped via process_ciu_reach when boy is present.
      - Units 22/23: retained only when required co-occurring units are present.

    Saves a per-file pickle DataFrame to `dir/dataframes/` and appends a CIU_seq
    column to all_df.
    """
    from striprtf.striprtf import rtf_to_text
    global lookup, unit, nodes, keyword_list, mapping

    dataframes_dir = os.path.join(dir, 'dataframes')
    os.makedirs(dataframes_dir, exist_ok=True)

    rtf_files = [f for f in all_df.file_dir if f.endswith('.rtf')]
    if rtf_files:
        from deepmultilingualpunctuation import PunctuationModel
        model = PunctuationModel()
    else:
        model = None

    required_22 = {9, 20, 12}
    required_23 = {1, 2, 6, 8, 18}

    entirelist = []
    pbar = tqdm(total=len(all_df.file_dir))

    for file in all_df.file_dir:
        mylist = []

        if file.endswith('.rtf'):
            with open(file, 'r', errors='ignore') as f:
                curr_speech = rtf_to_text(f.read()).lower()
            if not curr_speech.endswith('.'):
                curr_speech += '.'
            s = expand_repeats(curr_speech)
            s = _clean(s)
            s = re.sub(r'\s+', ' ', s).strip()
            sentences = re.split(r'[.,]', model.restore_punctuation(s))
        else:
            sentences = []
            with open(file, encoding='utf-8', errors='ignore') as text:
                curr_speech = ''
                for line in text:
                    if line.startswith('*PAR:'):
                        curr_speech += line
                    elif len(curr_speech) != 0 and not (line.startswith('%') or line.startswith('*') or line.startswith('@')):
                        curr_speech += line
                    elif len(curr_speech) != 0 and (line.startswith('%') or line.startswith('*') or line.startswith('@')):
                        s = re.sub('\*PAR:\t', ' ', curr_speech)
                        sentences.append(_clean(expand_repeats(s)))
                        curr_speech = ''
                    if ' '.join(str(line).split()) == '@G: Cat':
                        break

        for s in sentences:
            s = ' '.join([token.lemma_ for token in nlp(s)])
            try:
                templist = match_keyword_and_count(s)

                if 'cookie jar' in s:
                    templist = [x for i, x in enumerate(templist)
                                if not (x == 6 and i + 1 < len(templist) and templist[i + 1] == 7)]

                if 'fall' in s:
                    templist = [
                        x for i, x in enumerate(templist)
                        if x != 18 or (i > 0 and templist[i - 1] in {1, 2})
                    ]

                if 'reach' in s and len(templist) > 1:
                    templist = process_ciu_reach(templist)

                templist = [i for n, i in enumerate(templist) if i not in templist[:n]]

                if 22 in templist or 23 in templist:
                    has_required_22 = any(x in required_22 for x in templist)
                    has_required_23 = any(x in required_23 for x in templist)
                    templist = [x for x in templist
                                if (x != 22 or has_required_22) and (x != 23 or has_required_23)]

                mylist.extend(templist)
            except:
                print(f"Error processing utterance in {file}")

        new_list, unit_list, unit_labels, reference, quadrant = [], [], [], [], []
        entirelist.append(mylist)

        if not mylist:
            new_list, unit_list, unit_labels, reference, quadrant = [(0, 0)], 'null', 'null', 'null', 'null'
        else:
            for x in mylist:
                try:
                    templook = lookup[x - 1]
                    new_list.append(templook)
                    unit_list.append(unit[x - 1])
                    unit_labels.append(x)
                    reference.append(
                        'subject' if x <= 3 else 'place' if 3 < x <= 5 else 'object' if 5 < x <= 16 else 'action')
                    quadrant.append(
                        1 if templook[0] < 273 and templook[1] < 195 else
                        2 if templook[0] < 273 and templook[1] > 195 else
                        3 if templook[0] > 273 and templook[1] < 195 else 4)
                except:
                    print("Fail to look up: {:d}".format(x))
                    pass

        df = pd.DataFrame(new_list, columns=['x', 'y'])
        df['order'] = np.arange(len(df)) + 1
        df['unit'], df['unit #'], df['reference'], df['quadrant'] = unit_list, unit_labels, reference, quadrant
        filename = os.path.splitext(os.path.basename(file))[0]
        df.to_pickle(os.path.join(dataframes_dir, filename + '.pkl'))
        pbar.update(1)

    all_df['CIU_seq'] = entirelist
    return all_df


def summarize_spatio_semantics(dir=None):
    """Compute graph-theoretic spatio-semantic features from per-file pickles.

    Reads each .pkl saved by extract_spatio_semantics_asu, constructs a directed
    multigraph where nodes are semantic units and edges are sequential transitions
    weighted by Euclidean distance between unit positions on the Cookie Theft image,
    and computes path, coverage, cycle, and quadrant features.

    Returns a DataFrame with one row per file and columns:
    file_id, total_path_distance, unique_nodes, nodes, self_cycles, sum_of_edges,
    sum_of_edges/unique_nodes, sum_of_edges/nodes, cycles/unique nodes, cycles/nodes,
    avg_x, avg_y, std_x, std_y, self_cycles_quadrants, cross_ratio_quadrants.
    """
    global lookup, unit, nodes, keyword_list, mapping

    quadrant = [0, 1, 2, 2, 2, 0, 0, 1, 3, 2, 2, 3, 2, 0, 3, 2, 0, 1, 2, 3, 0, 2, 2]
    colors = ['blue', 'green', 'red', 'yellow']
    quads = [0, 1, 2, 3]
    quadpos = [(136.5, 292.5), (136.5, 97.5), (409.5, 292.5), (409.5, 97.5)]
    finaldict = {}
    markov = np.zeros((23, 23))
    markovs = np.zeros((23, 1))
    dataframes_dir = os.path.join(dir, 'dataframes')

    def new_add_edge(G, a, b, weight):
        if (a, b) in G.edges:
            max_rad = max(x[2]['rad'] for x in G.edges(data=True) if sorted(x[:2]) == sorted([a, b]))
        else:
            max_rad = 0
        G.add_edge(a, b, rad=max_rad + 0.1, weight=weight)

    for file in os.listdir(dataframes_dir):
        temp = pd.read_pickle(os.path.join(dataframes_dir, file))
        name = file[:-4]

        grph = nx.MultiDiGraph()
        nodes = nodes
        grph.add_nodes_from(list(nodes))
        colorslist = [colors[quadrant[nodes.index(unitnum)]] for unitnum in dict.fromkeys(nodes)]

        eucdist = 0
        for x in np.arange(len(temp['order'])):
            if x != len(temp['order']) - 1:
                eucdist = distance.euclidean((temp.x[x], temp.y[x]), (temp.x[x + 1], temp.y[x + 1]))
                new_add_edge(grph, temp.unit[x], temp.unit[x + 1], eucdist)

        finaldict[name] = {}
        dup_nodes = temp.pivot_table(index=['unit'], aggfunc='size')
        dup_nodes_list = [x for x in list(dup_nodes) if x != 1]
        cycles = math.ceil(sum(dup_nodes_list) / 2)
        finaldict[name]['cycles'] = cycles

        edgesum = grph.size(weight='weight')
        uniq_nodes = len(set(temp.unit))
        all_nodes = len(temp.unit)
        selfcycles = sum(1 for e in grph.edges if e[0] == e[1])

        if uniq_nodes == 0:
            uefficiency = efficiency = uredundancy = redundancy = 0
        else:
            uefficiency = edgesum / uniq_nodes
            efficiency = edgesum / all_nodes
            uredundancy = cycles / uniq_nodes
            redundancy = cycles / all_nodes

        finaldict[name].update({
            'file_id': re.split(r'_|\.', file)[0].lower(),
            'total_path_distance': edgesum,
            'unique_nodes': uniq_nodes,
            'nodes': all_nodes,
            'self_cycles': selfcycles,
            'sum_of_edges': edgesum,
            'sum_of_edges/unique_nodes': uefficiency,
            'sum_of_edges/nodes': efficiency,
            'cycles/unique nodes': uredundancy,
            'cycles/nodes': redundancy,
            'avg_x': mean(temp['x']),
            'avg_y': mean(temp['y']),
            'std_x': np.std(temp['x']),
            'std_y': np.std(temp['y'])
        })

        if temp['unit #'][0] != 'null':
            markovs[temp['unit #'][0] - 1] += 1
            for f in range(0, len(temp) - 1):
                markov[temp['unit #'][f] - 1, temp['unit #'][f + 1] - 1] += 1

        for n in range(0, len(nodes)):
            grph.nodes[nodes[n]]['pos'] = tuple(np.subtract((lookup[n][0] * 2, 390), lookup[n]))

        grph2 = nx.MultiDiGraph()
        grph2.add_nodes_from(quads)
        for x in np.arange(len(temp['order'])):
            if x != len(temp['order']) - 1:
                new_add_edge(grph2, quadrant[int(temp['unit #'][x]) - 1],
                             quadrant[int(temp['unit #'][x + 1]) - 1], weight=eucdist)
        for n in range(0, len(quads)):
            grph2.nodes[quads[n]]['pos'] = quadpos[n]

        selfcycles2 = sum(1 for e in grph2.edges if e[0] == e[1])
        finaldict[name]['self_cycles_quadrants'] = selfcycles2
        cross = sum(1 for e in grph2.edges if e[0] != e[1])
        finaldict[name]['cross_ratio_quadrants'] = 0 if selfcycles2 == 0 else cross / selfcycles2

    df = pd.DataFrame(finaldict).T
    df.reset_index(drop=True, inplace=True)
    return df


def plot_spatio_semantics(dir=None):
    """Plot the spatio-semantic transition graphs for each transcript in `dir`.

    Reads each .pkl produced by extract_spatio_semantics_asu and draws two directed
    multigraphs per transcript: a unit-level graph (23 nodes) and a quadrant-level
    graph (4 nodes). Nodes are colored by quadrant (blue/green/red/yellow).

    Figures are saved to:
      <dir>/plots/            — unit-level graph per transcript
      <dir>/quadrant_plots/   — quadrant-level graph per transcript
    """
    dataframes_dir = os.path.join(dir, 'dataframes')
    plots_dir = os.path.join(dir, 'plots')
    quadrant_plots_dir = os.path.join(dir, 'quadrant_plots')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(quadrant_plots_dir, exist_ok=True)
    global lookup, unit, nodes, keyword_list, mapping

    quadrant = [0, 1, 2, 2, 2, 0, 0, 1, 3, 2, 2, 3, 2, 0, 3, 2, 0, 1, 2, 3, 0, 2, 2]
    colors = ['blue', 'green', 'red', 'yellow']
    quads = [0, 1, 2, 3]
    quadpos = {n: p for n, p in zip(quads, [(136.5, 292.5), (136.5, 97.5), (409.5, 292.5), (409.5, 97.5)])}

    def new_add_edge(G, a, b, weight):
        if (a, b) in G.edges:
            max_rad = max(x[2]['rad'] for x in G.edges(data=True) if sorted(x[:2]) == sorted([a, b]))
        else:
            max_rad = 0
        G.add_edge(a, b, rad=max_rad + 0.1, weight=weight)

    for file in os.listdir(dataframes_dir):
        temp = pd.read_pickle(os.path.join(dataframes_dir, file))
        name = file[:-4]

        grph = nx.MultiDiGraph()
        nodes = nodes
        grph.add_nodes_from(list(nodes))
        colorslist = [colors[quadrant[nodes.index(unitnum)]] for unitnum in dict.fromkeys(nodes)]

        eucdist = 0
        for x in np.arange(len(temp['order'])):
            if x != len(temp['order']) - 1:
                eucdist = distance.euclidean((temp.x[x], temp.y[x]), (temp.x[x + 1], temp.y[x + 1]))
                new_add_edge(grph, temp.unit[x], temp.unit[x + 1], eucdist)

        for n in range(0, len(nodes)):
            grph.nodes[nodes[n]]['pos'] = tuple(np.subtract((lookup[n][0] * 2, 390), lookup[n]))
        pos = nx.get_node_attributes(grph, 'pos')

        # Unit-level graph
        figure = plt.figure()
        figure.set_size_inches(16.38, 11.7)
        nx.draw_networkx_nodes(grph, pos, node_color=colorslist)
        for edge in grph.edges(data=True):
            nx.draw_networkx_edges(grph, pos=pos, edgelist=[(edge[0], edge[1])],
                                   connectionstyle=f'arc3, rad = {edge[2]["rad"]}')
        for n in nodes:
            pos[mapping[n]] = pos.pop(n)
        grph = nx.relabel_nodes(grph, mapping)
        nx.draw_networkx_labels(grph, pos)
        plt.ylim(bottom=0, top=390)
        plt.xlim(left=0, right=546)
        plt.axis('off')
        plt.title(name)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{name}.png'), bbox_inches='tight', dpi=150)
        plt.close()

        # Quadrant-level graph (skipped for empty transcripts)
        if temp['unit #'][0] != 'null':
            grph2 = nx.MultiDiGraph()
            grph2.add_nodes_from(quads)
            for x in np.arange(len(temp['order'])):
                if x != len(temp['order']) - 1:
                    new_add_edge(grph2, quadrant[int(temp['unit #'][x]) - 1],
                                 quadrant[int(temp['unit #'][x + 1]) - 1], weight=eucdist)

            figure2 = plt.figure()
            figure2.set_size_inches(8, 6)
            nx.draw_networkx_nodes(grph2, quadpos, node_color=colors)
            for edge in grph2.edges(data=True):
                nx.draw_networkx_edges(grph2, pos=quadpos, edgelist=[(edge[0], edge[1])],
                                       connectionstyle=f'arc3, rad = {edge[2]["rad"]}')
            nx.draw_networkx_labels(grph2, quadpos)
            plt.ylim(bottom=0, top=390)
            plt.xlim(left=0, right=546)
            plt.axis('off')
            plt.title(f'{name} — quadrants')
            plt.tight_layout()
            plt.savefig(os.path.join(quadrant_plots_dir, f'{name}.png'), bbox_inches='tight', dpi=150)
            plt.close()
