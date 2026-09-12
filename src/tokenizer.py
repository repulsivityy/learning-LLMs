from collections import defaultdict


def get_pair_counts(word_freqs):
    pair_counts = defaultdict(int)
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += freq
    return pair_counts


def merge_pair(pair, word_freqs):
    new_word_freqs = {}
    for word, freq in word_freqs.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(word[i] + word[i + 1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word_freqs[tuple(new_word)] = freq
    return new_word_freqs


def train_bpe(word_freqs, num_merges):
    word_freqs = dict(word_freqs)
    merges = []
    for step in range(num_merges):
        pair_counts = get_pair_counts(word_freqs)
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=pair_counts.get)
        word_freqs = merge_pair(best_pair, word_freqs)
        merges.append(best_pair)
        print(f"Merge {step + 1}: {best_pair}  (count={pair_counts[best_pair]})")
    return word_freqs, merges


def get_word_tokens(word, merges):
    """Apply a learned merge list, in order, to a single word."""
    tokens = list(word) + ['</w>']
    for pair in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens


def encode(text, merges):
    """Whitespace-split text into words, then BPE-tokenize each word."""
    all_tokens = []
    for word in text.strip().split():
        all_tokens.extend(get_word_tokens(word, merges))
    return all_tokens


def decode(tokens):
    """Join tokens back into text."""
    text = ''.join(tokens).replace('</w>', ' ')
    return text.strip()
