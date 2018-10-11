#!/usr/bin/env python
import sys
sys.path.append('..')
import numpy as np
import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from cbow import CBOW
from common.util import create_contexts_target


def main():
    # ハイパーパラメータの設定
    window_size = 5
    hidden_size = 100
    batch_size = 100
    max_epoch = 10


if __name__ == '__main__':
    main()
