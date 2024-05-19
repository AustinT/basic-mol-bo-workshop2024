import argparse
import heapq
import json
import random
from typing import Callable

from mol_ga.mol_libraries import random_zinc
from tdc import Oracle


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

    # Run all tasks
    if args.method == "screening":
        random_smiles = random_zinc(size=2 * args.budget, rng=rng)  # double the budget to remove duplicates
        random_smiles = list(dict.fromkeys(random_smiles))[: args.budget]  # remove duplicates
        task(random_smiles)
    elif args.method == "tanimoto_gpbo":
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Save results
    results = {f"top{k}-AUC": auc_topk(task.cache, k=k, budget=args.budget) for k in [1, 10, 100]}
    with open(args.out_file, "w") as f:
        json.dump(results, f, indent=2)
    print("### END OF SCRIPT ###")
