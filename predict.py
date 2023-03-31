import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['tail', 'lysin'], help='VHH type: VHH-tail or VHH-lysin', required=True)
    parser.add_argument('--cds', help='Coding DNA sequences path (FASTA format)', required=True)
    parser.add_argument('--prot', help='Protein sequences path (FASTA format)', required=True)
    parser.add_argument('--out', help='Output path', default='results.csv')
    parser.add_argument('--precision', help='Precision', choices=['95', '84', '69', '-1'], default=-1)
    parser.add_argument('--bs', help='Batch size', default='32')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    from viral_host_hunter import ViralHostHnter
    vhh = ViralHostHnter(vhh_type=args.type, batch_size=int(args.bs))
    vhh.predict(dna_file=args.cds, protein_file=args.prot, precision=int(args.precision), output_file=args.out)
