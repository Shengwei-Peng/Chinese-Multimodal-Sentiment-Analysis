import torch
from argparse import ArgumentParser
from utils import extracted_features, split_data, Train, validation, visualization
from model import Feature_Fusion_Network

def parse_args():

    parser = ArgumentParser()
    parser.add_argument(
        "--processed_data", 
        type=str, 
        default=None,
        help="If you already have processed data (.pt, .pth), you can load it directly and skip the pre-processing steps."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-4,
    )
    parser.add_argument(
        "--eposhs", 
        type=int, 
        default=1,
    )
    parser.add_argument(
        "--early_stop", 
        type=int, 
        default=100,
    )
    parser.add_argument(
        "--model_save_to", 
        type=str, 
        default="model.pth",
    )
    parser.add_argument(
        "--regression", 
        action="store_true", 
    )
    args = parser.parse_args()
    
    return args


def main():
    torch.cuda.empty_cache()
    args = parse_args()

    data = torch.load(args.processed_data) if args.processed_data else extracted_features(
        format_path="./formatted",
        text_model="bert-base-chinese",
        audio_model="MIT/ast-finetuned-audioset-10-10-0.4593",
        vision_model="microsoft/xclip-base-patch32",
        data_save_to="processed_data.pt"
    )

    num_classes = 1 if args.regression else max(data["train"]["label_c"])+1
    loss_function = torch.nn.MSELoss() if args.regression else torch.nn.CrossEntropyLoss()

    train_loader, valid_loader, test_loader = split_data(data=data, batch_size=args.batch_size)

    model = Feature_Fusion_Network(
        t_in=data["train"]["text"][0].shape, 
        a_in=data["train"]["audio"][0].shape, 
        v_in=data["train"]["vision"][0].shape, 
        num_classes=num_classes,
        )
    
    model, history = Train(
        model, 
        loss_function, 
        train_loader, 
        valid_loader, 
        args.lr, 
        args.eposhs, 
        args.early_stop,
        args.model_save_to, 
        args.regression
        )

    model = torch.load(args.model_save_to)
    _, true, pred = validation(model, loss_function, test_loader, args.regression)
    visualization(history, true, pred, args.regression)

if __name__ == "__main__":
    main()
