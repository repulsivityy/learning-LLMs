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
