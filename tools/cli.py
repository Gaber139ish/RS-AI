import argparse

from tools.trainer import run_multithreaded_training
from tools.federated import main as federated_main


def main():
    parser = argparse.ArgumentParser(description="RS-AI CLI")
    sub = parser.add_subparsers(dest='cmd')

    t = sub.add_parser('train', help='Run trainer')
    t.add_argument('--seconds', type=int, default=8)

    c = sub.add_parser('chain', help='Run federated chain simulation')

    args = parser.parse_args()
    if args.cmd == 'train':
        import tomllib as toml_loader
        with open('configs/rs-config.toml', 'rb') as f:
            cfg = toml_loader.load(f)
        run_multithreaded_training(cfg, duration_seconds=args.seconds)
    else:
        federated_main()


if __name__ == '__main__':
    main()