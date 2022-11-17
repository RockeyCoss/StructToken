import argparse
import os
import os.path as osp
import wget
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--keys', nargs='+', help='The keys of the the checkpoints')
parser.add_argument('--folder', default='checkpoints', help='The folder to save the checkpoints')
parser.add_argument('--ckpt-names', nargs='+', help='The save name of the checkpoints')
parser.add_argument('--all', action='store_true', help='download all arguments')

args = parser.parse_args()
ckpt_dict = {
    "struct-token-cse_vit-b_ade20k": "https://github.com/RockeyCoss/StructToken/releases/download/v0.1.0/struct-token"
                                     "-cse_vit-b_ade20k.pth",
    "struct-token-cse_vit-s_ade20k": "https://github.com/RockeyCoss/StructToken/releases/download/v0.1.0/struct-token"
                                     "-cse_vit-s_ade20k.pth",
    "struct-token-cse_vit-t_ade20k": "https://github.com/RockeyCoss/StructToken/releases/download/v0.1.0/struct-token"
                                     "-cse_vit-t_ade20k.pth",
    "struct-token-pwe_vit-b_ade20k": "https://github.com/RockeyCoss/StructToken/releases/download/v0.1.0/struct-token"
                                     "-pwe_vit-b_ade20k.pth",
    "struct-token-pwe_vit-l_ade20k": "https://github.com/RockeyCoss/StructToken/releases/download/v0.1.0/struct-token"
                                     "-pwe_vit-l_ade20k.pth",
    "struct-token-pwe_vit-s_ade20k": "https://github.com/RockeyCoss/StructToken/releases/download/v0.1.0/struct-token"
                                     "-pwe_vit-s_ade20k.pth",
    "struct-token-pwe_vit-t_ade20k": "https://github.com/RockeyCoss/StructToken/releases/download/v0.1.0/struct-token"
                                     "-pwe_vit-t_ade20k.pth",
    "struct-token-sse_vit-b_ade20k": "https://github.com/RockeyCoss/StructToken/releases/download/v0.1.0/struct-token"
                                     "-sse_vit-b_ade20k.pth",
    "struct-token-sse_vit-l_ade20k": "https://github.com/RockeyCoss/StructToken/releases/download/v0.1.0/struct-token"
                                     "-sse_vit-l_ade20k.pth",
    "struct-token-sse_vit-s_ade20k": "https://github.com/RockeyCoss/StructToken/releases/download/v0.1.0/struct-token"
                                     "-sse_vit-s_ade20k.pth",
    "struct-token-sse_vit-t_ade20k": "https://github.com/RockeyCoss/StructToken/releases/download/v0.1.0/struct-token"
                                     "-sse_vit-t_ade20k.pth",
}

if args.keys is not None and args.ckpt_names is not None:
    assert len(args.keys) == len(args.ckpt_names), \
        'The length of --keys shouble equal to that of --ckpt-names'
assert (args.keys is not None) or args.all, \
    'You should either use --key to provide the keys of the checkpoints you want to download or use --all to download ' \
    'all checkpoints '
assert not (args.keys is not None and args.all), \
    'You should either use --key to provide the keys of the checkpoints you want to download or use --all to download ' \
    'all checkpoints '
if args.ckpt_names and args.all:
    assert len(args.keys) == len(ckpt_dict), \
    'If you want to download '

save_folder = args.folder
os.makedirs(save_folder, exist_ok=True)

if args.all:
    for key, url in tqdm(list(ckpt_dict.items())):
        wget.download(url=url, out=osp.join(save_folder, f'{key}.pth'))
else:
    if args.ckpt_names is not None:
        names = args.ckpt_names
    else:
        names = [f'{key}.pth' for key in args.keys]
    for key, name in tqdm(list(zip(args.keys, names))):
        wget.download(url=ckpt_dict[key],
                      out=osp.join(save_folder, name))



