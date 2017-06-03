from collections import Counter
import os

def get_data_for_path(path):
    counter = Counter()
    raw_data = []
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            with open('%s/%s' % (path, filename)) as f:
                file_tokens = []
                for line in f:
                    tokens = line.split()
                    for token in tokens:
                        counter[token] += 1
                        file_tokens.append(token)
                raw_data.append(file_tokens)
    return counter, raw_data
