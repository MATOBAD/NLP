#!/usr/bin/env python
# coding: utf-8
import sys
import argparse
sys.path.append('..')
from rnnlm_gen import RnnlmGen
from dataset import ptb


def main(word):
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    vocab_size = len(word_to_id)
    corpus_size = len(corpus)

    model = RnnlmGen()

    # start文字とskip文字の設定
    start_word = word
    start_id = word_to_id[start_word]
    skip_words = ['N', '<unk>', '$']
    skip_ids = [word_to_id[w] for w in skip_words]

    # 文章生成
    word_ids = model.generate(start_id, skip_ids=skip_ids)
    txt = ' '.join([id_to_word[i] for i in word_ids])
    txt = txt.replace(' <eos>', '.\n')
    print(txt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_word', default='you')
    args = parser.parse_args()
    word = args.start_word
    main(word)
