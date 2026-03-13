import os
import scipy
import numpy as np

def argparse():
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='extract representations from alphafold output'
    )
    parser.add_argument('--data-dir', default='/home/gpux1/ccd/npy_for_spare', type=str, help='path to alphafold data')
    parser.add_argument('--out-dir', default='/home/gpux1/ccd/spare_representations', type=str, help='output path')
    
    args = parser.parse_args()

    return args

def main(args):

    print(args)

    for fasta in os.listdir(args.data_dir):
        fname = [i for i in os.listdir(os.path.join(args.data_dir, fasta)) if i.startswith('result')][0]
        result = np.load(os.path.join(args.data_dir, fasta, fname), allow_pickle=True)
        rep = result['single'].mean(axis=0)
        dmax, dmin = rep.max(), rep.min()
        rep = (rep - dmin) / (dmax - dmin)
        rep = (rep - 0.5) * 2
        np.save(os.path.join(args.out_dir, fasta + '.npy'), rep)
    
    print('Representations extracted.')

if __name__ == '__main__':

    args = argparse()
    main(args)