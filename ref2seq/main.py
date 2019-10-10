import argparse
from train import trainIters
from evaluate import runTest
from load import Voc

def parse():
    parser = argparse.ArgumentParser(description='decoder rnn')
    parser.add_argument('-tr', '--train', help='Train the model with corpus')
    parser.add_argument('-te', '--test', help='Test the saved model')
    parser.add_argument('-l', '--load', help='Load the model and train')
    parser.add_argument('-c', '--corpus', help='Test the saved model with vocabulary of the corpus')
    parser.add_argument('-r', '--reverse', action='store_true', help='Reverse the input sequence')
    parser.add_argument('-f', '--filter', action='store_true', help='Filter to small training data set')
    parser.add_argument('-i', '--input', action='store_true', help='Test the model by input the sentence')
    parser.add_argument('-it', '--iteration', type=int, default=100000, help='Train the model with it iterations')
    parser.add_argument('-e', '--epoch', type=int, default=20, help='Train the model with e epochs')
    parser.add_argument('-dm', '--dmax', type=int, default=10, help='Max number of documents')
    parser.add_argument('-sm', '--smax', type=int, default=20, help='Max number of words in each document')
    parser.add_argument('-p', '--print', type=int, default=100, help='Print every p batches')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-la', '--layer', type=int, default=2, help='Number of layers in encoder and decoder')
    parser.add_argument('-hi', '--hidden', type=int, default=512, help='Hidden size in encoder and decoder')
    parser.add_argument('-be', '--beam', type=int, default=10, help='Hidden size in encoder and decoder')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('-endp', '--encoder_dropout', type=float, default=0.5, help='Encoder dropout rate')
    parser.add_argument('-dedp', '--decoder_dropout', type=float, default=0.2, help='Decoder dropout rate')
    parser.add_argument('-s', '--save', type=float, default=10000, help='Save every s iterations')
    parser.add_argument('-sa', '--sample', action='store_true', help='Use sampling to generation')

    args = parser.parse_args()
    return args

def parseFilename(filename, test=False):
    filename = filename.split('/')
    dataType = filename[-1][:-4] # remove '.tar'
    parse = dataType.split('_')
    print(filename)
    layers, hidden, _ = filename[-2].split('_')
    n_layers = int(layers.split('-')[0])
    hidden_size = int(hidden)
    return n_layers, hidden_size

def run(args):
    reverse, fil, n_epoch, print_every, learning_rate, n_layers, hidden_size, batch_size, beam_size, input = args.reverse, args.filter, args.epoch, args.print, args.learning_rate, args.layer, args.hidden, args.batch_size, args.beam, args.input

    if args.train and not args.load:
        trainIters(args, args.train, reverse, n_epoch, learning_rate, batch_size,
                    n_layers, hidden_size, print_every)
    elif args.load:
        n_layers, hidden_size = parseFilename(args.load)
        trainIters(args, args.train, reverse, n_epoch, learning_rate, batch_size,
                    n_layers, hidden_size, print_every, loadFilename=args.load)
    # load parameters from filename
    elif args.test: 
        n_layers, hidden_size  = parseFilename(args.test, True)
        runTest(args, n_layers, hidden_size, reverse, args.test, beam_size, batch_size, input, args.corpus)


if __name__ == '__main__':
    args = parse()
    run(args)
