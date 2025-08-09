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

from chain.node import FederatedChain


def main():
    config = load_toml('configs/rs-config.toml')
    base = 'data/chain'
    os.makedirs(base, exist_ok=True)
    num_nodes = int(config.get('chain', {}).get('num_nodes', 3))
    rounds = int(config.get('chain', {}).get('rounds', 5))
    fc = FederatedChain(config, base_dir=base, num_nodes=num_nodes)
    fc.run(rounds=rounds)


if __name__ == '__main__':
    main()