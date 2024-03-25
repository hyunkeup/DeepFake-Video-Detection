import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='../dataset/test_videos/', type=str, help='DFDC origin dataset path.')
    parser.add_argument('--sub_folders', default=["dfdc_train_part_0/"], nargs="+",
                        help='DFDC origin dataset subfolders.')
    parser.add_argument('--result_path', default='./results', type=str, help='Result save path.')

    return parser.parse_args()
