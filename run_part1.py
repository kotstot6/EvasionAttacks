
from itertools import product
import os
import csv
import argparse

parser = argparse.ArgumentParser(description='Run Part 1')
parser.add_argument('--sbatch', action='store_true', help='run parameter grid on sbatch')
parser.set_defaults(sbatch=False)
args = parser.parse_args()

# Prepare figures directory
if os.path.isdir('figures'):
    os.system('rm -rf figures')

os.mkdir('figures')

with open('part1_metrics.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Setting', 'Epoch', 'Accuracy'])

# String of args for single use
single_params = '--save_figure --save_model --pooling avg --activation relu --dropout 0.2 --lr 0.001 --weight_decay 0.0001 --batch_size 32 --seed 1'

# Grid of hyperparameters for sbatch
grid = {
    'pooling' : ['avg', 'max'],
    'activation' : ['tanh', 'relu'],
    'dropout' : [0.1, 0.2, 0.3],
    'lr' : [5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
    'weight_decay' : [1e-4, 1e-3, 1e-2],
    'batch_size' : [32],
    'seed' : [1]
}

# Utility function
def make_sbatch_params(grid):

    trials = [ { p : t for p, t in zip(grid.keys(), trial) }
                    for trial in list(product(*grid.values())) ]

    def trial_to_args(trial):
        arg_list = ['--' + param + ' ' + str(val) if type(val) != type(True)
                else '--' + param if val else '' for param, val in trial.items()]
        return ' '.join(arg_list)

    sbatch_params = [trial_to_args(trial) for trial in trials]

    return sbatch_params

sbatch_params = make_sbatch_params(grid)

if args.sbatch:

    print(len(sbatch_params), 'jobs will be submitted.')

    for params in sbatch_params:
        os.system('sbatch run_part1.sh \'' + params + '\'')

else:

    print('Interactive mode.')
    os.system('python3 part1.py ' + single_params)
