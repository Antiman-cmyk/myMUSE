from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division, absolute_import, print_function

from fasttext import load_model
import argparse
import errno

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print fasttext .vec file to stdout from .bin file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model to use",
    )
    parser.add_argument("--vec_pt", type=str, default="", help="vec path to save")
    args = parser.parse_args()

    f = load_model(args.model)
    words = f.get_words()
    words, freqs = f.get_words(include_freq=True)
    with open(args.vec_pt, 'w', encoding='utf-8') as vec:
        vec.write(str(len(words)) + " " + str(f.get_dimension()) + '\n')
        for w in words:
            v = f.get_word_vector(w)
            vec.write(w + ' ' + ' '.join(list(map(str, v))) + '\n')

    # cnt = 0
    # print(str(len(words)) + " " + str(f.get_dimension()))
    # for freq in freqs:
    #     print(freq)
    # for w in words:
    #     v = f.get_word_vector(w)
        # print(w + ' ' + str(v).strip('[]'))
        # vstr = ""
        # for vi in v:
        #     vstr += " " + str(vi)
        # try:
        #     # vstr = str(freqs[cnt]) + " " + vstr
        #     print(w + vstr)
        # except IOError as e:
        #     if e.errno == errno.EPIPE:
        #         pass
        # cnt += 1
