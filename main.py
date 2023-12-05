import torch
from argparse import ArgumentParser
from utils import format_CH_SIMS, extracted_features

def parse_args():

    parser = ArgumentParser()
    parser.add_argument(
        "--data_path", 
        type=str, 
        default=None,
        help="If you already have processed data (.pt, .pth), you can load it directly and skip the pre-processing steps."
    )
    args = parser.parse_args()
    
    return args


def main():
    torch.cuda.empty_cache()
    args = parse_args()

    if args.data_path:
        data = torch.load(args.data_path)
    else:
        format_CH_SIMS()
        data = extracted_features(
            data_path="formatted", 
            text_model="bert-base-chinese",
            audio_model="MIT/ast-finetuned-audioset-10-10-0.4593",
            vision_model="microsoft/xclip-base-patch32",
            data_save_to="processed_data.pt"
            )

if __name__ == "__main__":
    main()
