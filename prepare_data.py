"""Convert caller-assigned UTF-8 train, validation and optional test directories into JSONL."""
import argparse
import json
from pathlib import Path
from train import read_documents


def collect(directory, split, source, license_name):
    rows = []
    for path in sorted(directory.rglob('*.txt')):
        relative = path.relative_to(directory)
        # The caller controls grouping: a top-level directory is one source group.
        group = relative.parts[0] if len(relative.parts) > 1 else relative.stem
        rows.append({'id': f'{split}/{relative.as_posix()}', 'group': group,
                     'text': path.read_text(encoding='utf-8'), 'source': source,
                     'license': license_name, 'source_path': relative.as_posix()})
    if not rows:
        raise ValueError(f'No .txt documents found in {directory}')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', type=Path, required=True)
    parser.add_argument('--validation-dir', type=Path, required=True)
    parser.add_argument('--test-dir', type=Path, help='untouched final holdout')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', required=True, help='Provenance identifier or URL; no automatic download')
    parser.add_argument('--license', required=True, help='The actual terms or your ownership statement')
    args = parser.parse_args()
    directories = [('train', args.train_dir), ('validation', args.validation_dir)]
    if args.test_dir:
        directories.append(('test', args.test_dir))
    records = {split: collect(directory, split, args.source, args.license)
               for split, directory in directories}
    args.output.mkdir(parents=True, exist_ok=False)
    for split, rows in records.items():
        (args.output / f'{split}.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    rows = read_documents(args.output / 'train.jsonl', args.output / 'validation.jsonl',
                          args.output / 'test.jsonl' if args.test_dir else None)
    print(json.dumps({'documents': len(rows), 'output': str(args.output),
                      'group_policy': 'top-level source folder or standalone filename stem',
                      'limitation': 'exact duplicates and declared group overlap checked; near duplicates require a separate audit'}, indent=2))


if __name__ == '__main__': main()
