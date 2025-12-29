import argparse
from dinov2.data.datasets import ImageNet


def parse_args():
    parser = argparse.ArgumentParser(description="Dump ImageNet extra for all splits.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="dentvista_2d_pretrain_dataset",
        help="Dataset name used to build default root/extra paths if not provided.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Dataset root path. If not set, defaults to /dataset_root/<dataset_name>",
    )
    parser.add_argument(
        "--extra",
        type=str,
        default=None,
        help="Extra path. If not set, defaults to /dataset_root/<dataset_name>/Extra",
    )
    parser.add_argument("--train_num", type=int, default=1745968)
    parser.add_argument("--val_num", type=int, default=2400)
    parser.add_argument("--test_num", type=int, default=2400)
    return parser.parse_args()


def main():
    args = parse_args()

    root = args.root or f"/dataset_root/{args.dataset_name}"
    extra = args.extra or f"{root}/Extra"

    for split in ImageNet.Split:
        dataset = ImageNet(
            split=split,
            root=root,
            extra=extra,
            train_num=args.train_num,
            val_num=args.val_num,
            test_num=args.test_num,
        )
        dataset.dump_extra()


if __name__ == "__main__":
    main()
