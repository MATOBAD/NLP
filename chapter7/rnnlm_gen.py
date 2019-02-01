#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
from common.functions import softmax
from chapter6.rnnlm import Rnnlm
from chapter6.better_rnnlm import BetterRnnlm


class RnnlmGen(Rnnlm):
    def generate(self, start_id, skip_ids=None, sample_size=100):
        '''
        start_id: 最初に与える単語ID
        sample_size: サンプリングする単語の数
        skip_ids: 指定された単語IDがサンプリングされないようにするリスト
        '''
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)

            # 各単語のスコアの出力
            score = self.predict(x)

            # softmax関数で正規化
            p = softmax(score.flatten())

            sampled = np.random.choice(len(p), size=1, p=p)

            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

        return word_ids
