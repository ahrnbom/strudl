""" Script for removing all but the best weights for a training run, to save
    several gigabytes of disk storage
"""

from glob import glob
import os.path
from os import remove
import numpy as np
import click

from folder import runs_path

@click.command()
@click.option("--dataset", default="sweden2", help="Name of dataset")
@click.option("--run", default="default", help="Name of training run")
def main(dataset, run):
    cleanup(dataset, run)

def cleanup(dataset, run):
    weights_files = glob(os.path.join(runs_path, '{dataset}_{run}'.format(dataset=dataset,run=run), 'checkpoints', '*.hdf5'))
    weights_files_loss = np.array([float(wf.split('-')[-1].replace('.hdf5', '')) for wf in weights_files])
    weights_file = weights_files[np.argmin(weights_files_loss)]
    
    for wf in weights_files:
        if not (wf == weights_file):
            remove(wf)
            
if __name__ == '__main__':
    main()


        


