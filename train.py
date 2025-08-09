import os

try:
    import tomllib as toml_loader  # Python 3.11+
    def load_toml(path):
        with open(path, 'rb') as f:
            return toml_loader.load(f)
except Exception:
    import toml as toml_loader  # type: ignore
    def load_toml(path):
        return toml_loader.load(path)

from tools.trainer import run_multithreaded_training


def main():
    config = load_toml('configs/rs-config.toml')
    os.makedirs('data/logs', exist_ok=True)
    os.makedirs('data/sponge', exist_ok=True)
    os.makedirs('data/checkpoints', exist_ok=True)

    runtime = int(config.get('orchestrator', {}).get('runtime_seconds', 8))
    run_multithreaded_training(config, duration_seconds=runtime)


if __name__ == '__main__':
    main()