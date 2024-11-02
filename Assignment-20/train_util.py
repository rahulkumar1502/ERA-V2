import unicodedata
import regex as re

def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def encode(text, merges, compiled_pattern):
    """Encoding that ignores any special tokens."""
    # split text into chunks of text by categories defined in regex pattern
    text_chunks = re.findall(compiled_pattern, text)
    # all chunks of text are encoded separately, then results are joined
    ids = []
    for chunk in text_chunks:
        chunk_bytes = chunk.encode("utf-8") # raw bytes
        chunk_ids = _encode_chunk(chunk_bytes, merges)
        ids.extend(chunk_ids)
    return ids

def decode(ids, vocab):
    # given ids (list of integers), return Python string
    part_bytes = []
    for idx in ids:
        if idx in vocab:
            part_bytes.append(vocab[idx])
        else:
            raise ValueError(f"invalid token id: {idx}")
    text_bytes = b"".join(part_bytes)
    text = text_bytes.decode("utf-8", errors="replace")
    return text

def _encode_chunk(text_bytes, merges):
    # return the token ids
    # let's begin. first, convert all bytes to integers in range 0..255
    ids = list(text_bytes)
    while len(ids) >= 2:
        # find the pair with the lowest merge index
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        # subtle: if there are no more merges available, the key will
        # result in an inf for every single pair, and the min will be
        # just the first pair in the list, arbitrarily
        # we can detect this terminating case by a membership check
        if pair not in merges:
            break # nothing else can be merged anymore
        # otherwise let's merge the best pair (lowest merge index)
        idx = merges[pair]
        ids = merge(ids, pair, idx)
    return ids