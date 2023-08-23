import fasttext
import argparse

parser = argparse.ArgumentParser(
        description="Print fasttext .vec file to stdout from .bin file"
    )
parser.add_argument("--data_pt", type=str, default="", help="")
parser.add_argument("--model_pt", type=str, default="", help="")
parser.add_argument("--epoch", type=int, default=5, help="")
parser.add_argument("--mode", type=str, default="skipgram", help="skipgram or cbow")
parser.add_argument("--dim", type=int, default=300, help="word vec dim")
parser.add_argument("--lr", type=float, default=0.05, help="")
parser.add_argument("--thread", type=int, default=5, help="")
args = parser.parse_args()

# data_pt = '/dat01/laizhiquan/psl/Project/UNMT/transformers/cache/monolingual/WMT16/en-new'
# model_pt = '/dat01/laizhiquan/psl/Project/UNMT/fastText/models/wmt16/en.bin'
# data_pt = r'D:\slpan\unmt\MUSE\data\kk2en\kk_KZ\train'
# model_pt = r'D:\slpan\unmt\MUSE\data\kk2en\kk_KZ\kk_KZ.bin'
model = fasttext.train_unsupervised(args.data_pt, model=args.mode, dim=args.dim, epoch=args.epoch, lr=args.lr, thread=args.thread)
model.save_model(args.model_pt)
# model = fasttext.load_model(model_pt)
