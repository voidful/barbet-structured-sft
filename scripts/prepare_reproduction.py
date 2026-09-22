"""Download pinned inputs; never start training or substitute SFT for base weights."""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/sft_20260916'))
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--include-base',action='store_true',help='Requires existing access to the original base')
    mode.add_argument('--include-sft',action='store_true',help='Public SFT-only evaluation inputs')
    args=parser.parse_args()
    if args.include_base:
        snapshot_download('OpenFormosa/barbet-1b-base',revision='4dbb35216a32de59570a3f0d164804f25c82009d',
                          local_dir=args.root/'base')
    else:
        if (args.root/'base/model.safetensors').exists():
            parser.error('SFT-only preparation expects no base weights; use a separate output root.')
        snapshot_download('OpenFormosa/barbet-1b-structured-sft',revision='v0.1.0',
                          local_dir=args.root/'public_sft')
        snapshot_download('OpenFormosa/barbet-1b-structured-sft',revision='v0.1.0',
                          local_dir=args.root/'base',allow_patterns=['*token*','vocab.json','merges.txt','config.json'])
    snapshot_download('voidful/barbet-sft',repo_type='dataset',revision='0bf2389cbcd6e828af9df0676aed56abc61c4c2e',
                      local_dir=args.root/'dataset',allow_patterns=['data/*.parquet','code/**','README.md','LICENSE_DATA.md'])
    print(f'Inputs ready under {args.root}. No training has been launched.')


if __name__=='__main__':
    main()
