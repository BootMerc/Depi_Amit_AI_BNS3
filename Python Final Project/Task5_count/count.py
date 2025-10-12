def counts_unique(words):
    """
    words: list of words.

    Counts the unique words in a list.
    
    return: dictionary with unique words as keys and their counts as values.
    """
    unique_words = set(words)
    counts = {word: 0 for word in unique_words}
    for i in unique_words:
        for j in words:
            if i == j:
                counts[i] += 1
    return counts
words = ["Welcome", "Ali", "Hi", "Ali", "No", "Hi", "No", "Ali", "No", "Ali"]
print(counts_unique(words))