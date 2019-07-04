""" Script for removing all but the best weights for a training run, to save
    several gigabytes of disk storage
"""

from glob import glob
import os.path
import numpy as np
import click

from folder import runs_path

@click.command()
@click.option("--dataset", default="sweden2", help="Name of dataset")
@click.option("--run", default="default", help="Name of training run")
def main(dataset, run):
    cleanup(dataset, run)

def cleanup(dataset, run):
    weights_files = list((runs_path / "{}_{}".format(dataset,run) / "checkpoints").glob('*.hdf5'))
    weights_files_loss = np.array([float(wf.stem.split('-')[-1]) for wf in weights_files])
    weights_file = weights_files[np.argmin(weights_files_loss)]
    
    for wf in weights_files:
        if not (wf == weights_file):
            wf.unlink()
            
if __name__ == '__main__':
    main()


        


