"""
Reservoir-sample N reads from a (potentially huge) FASTA without loading it all.

Usage:
    python subsample_unclassified.py <input.fasta> <output.fasta> [N] [--seed S]

Examples:
    python subsample_unclassified.py SRR37006671_unclassified.fasta SRR37006671_unclassified_250k.fasta 250000
    python subsample_unclassified.py SRR37006671_unclassified.fasta out.fasta 250000 --seed 42

Memory usage stays bounded at ~N records regardless of input size.
"""
import argparse
import random
import sys


def iter_fasta(path):
    """Yield (header, sequence) tuples one at a time. Constant memory."""
    rid, seq_parts = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith('>'):
                if rid is not None:
                    yield rid, ''.join(seq_parts)
                rid = line
                seq_parts = []
            else:
                seq_parts.append(line)
        if rid is not None:
            yield rid, ''.join(seq_parts)


def reservoir_sample(iterable, k, rng):
    """Algorithm R: uniform sample of k items from a stream of unknown length."""
    reservoir = []
    for i, item in enumerate(iterable):
        if i < k:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = item
        if (i + 1) % 1_000_000 == 0:
            print(f'  scanned {i+1:,} reads...', file=sys.stderr)
    return reservoir, i + 1 if reservoir else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input_fasta')
    ap.add_argument('output_fasta')
    ap.add_argument('n', type=int, nargs='?', default=250_000,
                    help='Number of reads to sample (default 250000)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print(f'Reservoir sampling {args.n:,} reads from {args.input_fasta}...')
    sample, total = reservoir_sample(iter_fasta(args.input_fasta), args.n, rng)
    print(f'Scanned {total:,} reads total. Sampled {len(sample):,}.')

    if len(sample) < args.n:
        print(f'WARNING: input had fewer reads ({len(sample):,}) than requested ({args.n:,}).',
              file=sys.stderr)

    with open(args.output_fasta, 'w') as f:
        for rid, seq in sample:
            f.write(f'{rid}\n{seq}\n')
    print(f'Wrote {args.output_fasta}')


if __name__ == '__main__':
    main()
