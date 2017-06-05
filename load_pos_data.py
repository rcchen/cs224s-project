import nltk
import os

from src.datasets.utils import *
from src.utils.progbar import Progbar


"""Loads all of the essay-only POS data into text files under
var/data/pos/{split}/{speaker_id}.txt."""

def main():

    essay_path = 'var/data/essays'
    outfile_base_path = 'var/data/pos'

    paths = [ os.path.join(outfile_base_path, subdir) for subdir in ['', 'dev', 'train', 'dev/tokenized', 'train/tokenized']]

    for d in paths:
        if not os.path.isdir(d):
            os.mkdir(d)

    for split in ['dev', 'train']:

        target = 11000 if split == 'train' else 1100

        prog = Progbar(target=target)

        split_essay_path = os.path.join(essay_path, split, 'original')
        _, raw_data, speaker_ids = get_data_for_path(split_essay_path)

        outfile_path = os.path.join(outfile_base_path, split, 'tokenized')
        i = 0
        for data, speaker_id in zip(raw_data, speaker_ids):
            with open(os.path.join(outfile_path, '%s.txt' % str(speaker_id).zfill(5)), 'w') as f:
                pos_data = [a[1] for a in nltk.pos_tag(data)]
                f.write(' '.join(pos_data))
                prog.update(i + 1, [('Writing %s files' % split, 1)])
                i += 1


if __name__ == '__main__':
    main()
