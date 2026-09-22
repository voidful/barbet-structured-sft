"""Run the pinned public Barbet checkpoint without altering its parameter dtypes."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--example', default=None, help='Bundled example name, e.g. ocr_correction')
    inputs.add_argument('--messages', type=Path, help='JSON messages array or example object')
    inputs.add_argument('--prompt', help='Plain instruction; free-form capability remains weak')
    parser.add_argument('--model', default='OpenFormosa/barbet-1b-structured-sft')
    parser.add_argument('--revision', default='v0.1.1')
    parser.add_argument('--local-model', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-new-tokens', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.prompt is not None:
        messages = [dict(role='system',content='請遵照要求，忠實使用提供的文字資訊。'),
                    dict(role='user',content=args.prompt)]
        default_limit = 256
    else:
        path = args.messages or root/'examples'/((args.example or 'ocr_correction')+'.json')
        content = json.loads(path.read_text())
        messages = content['messages'] if isinstance(content,dict) else content
        default_limit = content.get('max_new_tokens',1024) if isinstance(content,dict) else 1024
    if not isinstance(messages,list) or not messages or messages[-1].get('role') != 'user':
        parser.error('Messages must be a nonempty list ending in a user message.')
    if any(m.get('role') not in ['system','user','assistant'] or not isinstance(m.get('content'),str) for m in messages):
        parser.error('Every message needs a supported role and string content.')
    limit = args.max_new_tokens if args.max_new_tokens is not None else default_limit
    if limit <= 0:
        parser.error('--max-new-tokens must be positive')
    if args.device == 'cpu':
        # Optional CUDA dependencies can probe hardware during imports.
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from huggingface_hub import snapshot_download
    folder = (args.local_model.resolve() if args.local_model else
              Path(snapshot_download(args.model,revision=args.revision)))
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(4)
    sys.path.insert(0,str(folder))
    from load_barbet import load_barbet, generate_tokens
    model, tokenizer = load_barbet(folder,device=args.device)
    chat = AutoTokenizer.from_pretrained(folder,trust_remote_code=True,local_files_only=True)
    ids = chat.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
    if len(ids)+limit > model.config.max_position_embeddings:
        parser.error('Prompt plus requested generation exceeds the configured context limit; input was not truncated.')
    generated = generate_tokens(model,ids,max_new_tokens=limit)
    eos = bool(generated and generated[-1]==model.config.eos_token_id)
    text = tokenizer.decode(generated[:-1] if eos else generated,skip_special_tokens=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        result = dict(model=args.model,resolved_revision=folder.name if not args.local_model else None,
                      local_model=str(folder) if args.local_model else None,
                      input_tokens=len(ids),generated_tokens=len(generated),eos=eos,
                      limit_reached=not eos and len(generated)==limit,decoding='greedy',prediction=text)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    main()
