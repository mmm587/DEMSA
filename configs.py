import argparse
import random

def seed(s):
    if isinstance(s, int):
        if 0 <= s <= 9999:
            return s
        else:
            raise argparse.ArgumentTypeError(
                "Seed must be between 0 and 2**32 - 1. Received {0}".format(s)
            )
    elif s == "random":
        return random.randint(0, 9999)
    else:
        raise argparse.ArgumentTypeError(
            "Integer value is expected. Recieved {0}".format(s)
        )


parser = argparse.ArgumentParser(description='Multimodal Sentiment Analysis')
parser.add_argument('--model', type=str, default="xlnet-base-cased", help='bert-base-uncased, xlnet-base-cased')
parser.add_argument('--name', type=str, default='diff', help='name of the model to use (Transformer, etc. ma)')
parser.add_argument('--no_cuda', action='store_true', default=False, help='do not use cuda')
parser.add_argument('--dataset', type=str, default='mosi', help='default: mosei/mosi/iemocap')
parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--seed", type=seed, default="random")
parser.add_argument("--learning_rate", type=float, default=1e-5, help='1e-5')
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--n_epochs", type=int, default=40)  
parser.add_argument("--train_batch_size", type=int, default=48)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
args = parser.parse_args()  


if args.dataset == 'mosi':
    # MOSI SETTING
    AUDIO_DIM = 74
    VISION_DIM = 47
    TEXT_DIM = 768
elif args.dataset == 'mosei':
    # MOSEI SETTING
    AUDIO_DIM = 74
    VISION_DIM = 35
    TEXT_DIM = 768