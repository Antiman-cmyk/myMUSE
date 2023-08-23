import argparse
import os
import sys

import numpy as np
import torch

import kenlm
from torch import nn

from src.utils import bool_flag, load_embeddings, normalize_embeddings
from src.evaluation.translator import Translator
from src.logger import create_logger


def is_zip(fname):
    fp = open(fname, "rb")
    magic = fp.read(2)
    fp.close()
    return magic == '\x1f\x8b'


def open_file(file_arg, allowNone=True):
    if not file_arg:
        if allowNone:
            return None
        else:
            raise IOError
    if isinstance(file_arg, str):
        if is_zip(file_arg):
            return gzip.GzipFile(file_arg)
        else:
            # return open(file_arg)
            # psl edit
            return open(file_arg, encoding='utf8')
    elif isinstance(file_arg, file):
        return file_arg


def load_input(filename, lowercase):
    input_stream = sys.stdin
    if filename is not None:
        input_stream = open_file(filename)
    input_sents = []
    for line in input_stream:
        if lowercase:
            line = line.lower()
        input_sents.append(line.rstrip().split())
    return input_sents


def initialize_exp(params):
    """
    Initialize experiment.
    """
    # initialization
    if getattr(params, 'seed', -1) >= 0:
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
        if params.cuda:
            torch.cuda.manual_seed(params.seed)

    # create logger
    logger = create_logger(os.path.join(params.exp_path, 'train.log'), vb=params.verbose)
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v in sorted(dict(vars(params)).items())))
    logger.info('The experiment will be stored in %s' % params.exp_path)
    return logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')

    # data
    parser.add_argument('--input', type=str, help='Source corpus to translate (default: stdin)')
    parser.add_argument('--output', type=str,
                        help='File name where the output translations will be written (default: stdout)')
    parser.add_argument('--input_lowercase', type=bool_flag, default=True, help='Lowercase the given input corpus')
    # psl
    parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
    parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")

    # embedding
    parser.add_argument('--src_emb', type=str, required=True, help='Source embeddings')
    parser.add_argument('--tgt_emb', type=str, required=True, help='Target embeddings')
    parser.add_argument('--emb_dim', type=int, required=True, help='Embedding dimension')
    parser.add_argument('--normalize_embeddings', type=str, default='', help='Normalize embeddings before translation')

    # search
    parser.add_argument('--max_vocab', type=int, default=100000,
                        help='Maximum source/target vocabulary size considered in search')
    parser.add_argument('--similarity_measure', type=str, default='csls',
                        help='Similarity measure used for nearest neighbor search (cosine|csls)')
    parser.add_argument('--csls_k', type=int, default=10,
                        help='Number of nearest neighbors for computing hub penalty term in csls')
    parser.add_argument('--unknown_words', type=str, default='copy',
                        help='Method to deal with unknown source word (copy|unk)')

    # lm integration
    parser.add_argument('--lm', type=str, help='Target language model file')
    parser.add_argument('--lm_scaling', type=float, default=0.1, help='Language model scaling')
    parser.add_argument('--lex_scaling', type=float, default=1.0, help='Lexicon model scaling')
    parser.add_argument('--beam_size', type=int, default=10, help='Beam size')
    parser.add_argument('--topk', type=int, default=100, help='Number of nearest neighbors considered')
    parser.add_argument('--emb_scaling', type=str, default='linear', help='(linear|sigmoid|softmax)')
    parser.add_argument('--softmax_temparature', type=float, default=1.0,
                        help='Temparature parameter for scaling softmax distribution')

    # others
    parser.add_argument('--cuda', type=bool_flag, default=True, help='Run on GPU')

    # parse parameters
    params = parser.parse_args()
    params.src_lang = 'src'
    params.tgt_lang = 'tgt'  # lang code can be arbitrary since we don't load dictionary files

    # check parameters
    assert os.path.isfile(params.src_emb)
    assert os.path.isfile(params.tgt_emb)

    # psl edit
    logger = initialize_exp(params)

    # load input (to translate)
    # print("loading input data...", file=sys.stderr)
    logger.info("loading input data...")  # psl
    input_sents = load_input(params.input, params.input_lowercase)  # CHECK: vocab?

    # load embeddings
    # print("loading embeddings...", file=sys.stderr)
    logger.info("loading embeddings...")  # psl
    src_dico, _src_emb = load_embeddings(params, source=True)  # 'dico' = word2id mappings
    src_emb = nn.Embedding(len(src_dico), params.emb_dim, sparse=True)
    src_emb.weight.data.copy_(_src_emb)

    tgt_dico, _tgt_emb = load_embeddings(params, source=False)
    tgt_emb = nn.Embedding(len(tgt_dico), params.emb_dim, sparse=True)
    tgt_emb.weight.data.copy_(_tgt_emb)

    if params.cuda:
        src_emb.cuda()
        tgt_emb.cuda()

    # normalize embeddings
    # print("normalizing embeddings...", file=sys.stderr)
    logger.info("normalizing embeddings...")  # psl
    params.src_mean = normalize_embeddings(src_emb.weight.data, params.normalize_embeddings)
    params.tgt_mean = normalize_embeddings(tgt_emb.weight.data, params.normalize_embeddings)

    # load lm
    lm = None
    if params.lm is not None:
        print("loading LM...", file=sys.stderr)
        lm = kenlm.LanguageModel(params.lm)

    # translate
    # print("translating...", file=sys.stderr)
    logger.info("translating...")
    translator = Translator(src_emb, tgt_emb, src_dico, tgt_dico, params)
    translator.corpus_translation(input_sents, lm)
