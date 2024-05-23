import argparse
import heapq
import json
import random
from typing import Callable

from tdc import Oracle

from tanimoto_gpbo import run_tanimoto_gpbo


class OracleWrapper:
    def __init__(self, oracle: Callable[[list[str]], list[float]]):
        self._oracle = oracle
        self.cache = {}

    def __call__(self, smiles: list[str]) -> list[float]:
        smiles_not_in_cache = [s for s in smiles if s not in self.cache]
        if smiles_not_in_cache:
            results = self._oracle(smiles_not_in_cache)
            for s, r in zip(smiles_not_in_cache, results):
                self.cache[s] = float(r)
        return [self.cache[s] for s in smiles]  # all SMILES will now be in the cache


def auc_topk(call_dict, k: int, budget: int) -> float:
    if k > budget:
        raise ValueError(f"k={k} is greater than the budget={budget}")
    items = list(call_dict.items())[:budget]

    # Find top k at every time step
    def key(x):
        return x[1]

    total = 0.0
    current_topk = []
    for i in range(budget):
        if i < len(items):
            current_topk.append(items[i])
        current_topk = heapq.nlargest(k, current_topk, key=key)
        kth_largest = key(current_topk[-1])
        total += kth_largest

    return total / budget


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--oracle", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--budget", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Random seed
    rng = random.Random(args.seed)

    # Create oracle
    task = OracleWrapper(oracle=Oracle(args.oracle))

    # Load SMILES bank
    with open("zinc.tab") as f:
        smiles_bank = [line.strip().strip('"') for line in f.readlines()]
        smiles_bank = smiles_bank[1:]  # remove header
        smiles_bank = list(set(smiles_bank))  # remove duplicates

    # Run all tasks
    if args.method == "screening":
        random_smiles = rng.sample(smiles_bank, args.budget)
        task(random_smiles)
    elif args.method == "tanimoto_gpbo":
        run_tanimoto_gpbo(oracle=task, smiles_bank=smiles_bank, rng=rng, oracle_budget=args.budget)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Save results
    results = {f"top{k}-AUC": auc_topk(task.cache, k=k, budget=args.budget) for k in [1, 10, 100]}
    with open(args.out_file, "w") as f:
        json.dump(results, f, indent=2)
    print("### END OF SCRIPT ###")
