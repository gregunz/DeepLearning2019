import argparse

import torch

from constants import NUM_EPOCHS, NUM_ROUNDS, SEED, AUX_LOSS_FACTOR
from evaluation import evaluate_model
from models import SimpleFC, SimpleCNN
from plot import plot_from_tensors

parser = argparse.ArgumentParser()
parser.add_argument('--models', action="store", dest="models", type=str, default='1234')
parser.add_argument('--epochs', action="store", dest="epochs", type=int, default=NUM_EPOCHS)
parser.add_argument('--rounds', action="store", dest="rounds", type=int, default=NUM_ROUNDS)
parser.add_argument('--seed', action="store", dest="seed", type=int, default=SEED)
parser.add_argument('--aux', action="store", dest="aux_loss_factor", type=float, default=AUX_LOSS_FACTOR)
args = parser.parse_args()

accuracies = []
losses = []
model_names = []


#  Evaluation method for each model with some fancy prints and the accumulations of the statistics
def evaluate(model, shortname, fullname, with_aux):
    shortname = shortname + ('_aux' if with_aux else '')
    s = '' if with_aux else 'out'
    print(f'Training "{fullname} ({shortname})" model with{s} auxiliary training loss')

    torch.manual_seed(args.seed)
    loss, acc = evaluate_model(
        model,
        num_epochs=args.epochs,
        num_rounds=args.rounds,
        with_aux_classes=with_aux,
        aux_loss_factor=args.aux_loss_factor
    )
    losses.append(loss)
    accuracies.append(acc)
    model_names.append(shortname)


def main():
    # Models evaluation
    if args.models is None or '1' in args.models:
        evaluate(SimpleFC, 'FC', 'Fully Connected', with_aux=False)
    if args.models is None or '2' in args.models:
        evaluate(SimpleFC, 'FC', 'Fully Connected', with_aux=True)
    if args.models is None or '3' in args.models:
        evaluate(SimpleCNN, 'CNN', 'Convolutional', with_aux=False)
    if args.models is None or '4' in args.models:
        evaluate(SimpleCNN, 'CNN', 'Convolutional', with_aux=True)

    # Plotting results
    if len(accuracies) > 0:
        plot_from_tensors(accuracies, losses, model_names, save_df=True)


if __name__ == '__main__':
    main()
