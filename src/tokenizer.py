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
