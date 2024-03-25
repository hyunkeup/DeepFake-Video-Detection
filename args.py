import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # Architecture
    parser.add_argument("--fusion", default="it", type=str, help="Fusion can be 'lt', 'it', or 'ia'")

    # Datasets
    parser.add_argument('--annotation_path', default='./annotations.txt', type=str, help='Annotation file path')
    parser.add_argument('--result_path', default='results', type=str, help='Result directory path')
    parser.add_argument("--n_threads", default=16, type=int, help="Number of threads for multi-thread loading")

    # Device
    parser.add_argument('--device', default='cuda', type=str,
                        help='Specify the device to run. Defaults to cuda, fallsback to cpu')

    # Hyperparameters
    parser.add_argument("--learning_rate", default=0.04, type=float, help="Learning rate")
    parser.add_argument('--n_epochs', default=100, type=int, help='Number of total epochs to run')
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size")

    # Running options
    parser.add_argument('--train', default=True, action=argparse.BooleanOptionalAction,
                        help='If false, training is not performed.')
    parser.add_argument('--test', default=True, action=argparse.BooleanOptionalAction,
                        help='If false, test is not performed.')
    parser.add_argument('--val', default=True, action=argparse.BooleanOptionalAction,
                        help='If false, test is not performed.')

    # Marlin
    parser.add_argument('--marlin_path', default=None, type=str,
                        help='Marlin path. If it is None, it will use online release.')

    return parser.parse_args()
