"""Script to find similar molecules to a target molecule."""

import argparse
import logging
import random
import sys

from mol_ga import default_ga
from mol_ga.mol_libraries import random_zinc
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity


def binary_similarity_objective(s1: str, s2: str):
    m1 = Chem.MolFromSmiles(s1)
    m2 = Chem.MolFromSmiles(s2)
    if m1 is None or m2 is None:
        return 0.0
    kwargs = dict(radius=2, useCounts=False)
    fp1 = AllChem.GetMorganFingerprint(m1, **kwargs)
    fp2 = AllChem.GetMorganFingerprint(m2, **kwargs)
    return TanimotoSimilarity(fp1, fp2)


def main(target_smiles: str, max_generations: int):
    rng = random.Random()

    def batch_objective(smiles_list):
        return [binary_similarity_objective(s1=target_smiles, s2=s) for s in smiles_list]

    ga_logger = logging.getLogger("mol_ga")
    ga_logger.setLevel(logging.DEBUG)
    ga_output = default_ga(
        starting_population_smiles=random_zinc(10_000, rng=rng) + [target_smiles],
        scoring_function=batch_objective,
        max_generations=max_generations,
        offspring_size=100,
        population_size=10_000,
        rng=rng,
        parallel=None,
        logger=ga_logger,
    )

    print("Top 100 similar molecules to the target molecule:")
    for s, score in sorted(ga_output.scoring_func_evals.items(), key=lambda x: x[1], reverse=True)[:100]:
        print(f"{score:.3f}\t\t\t{s}")

    print("\nLargest molecules with similarity 1.0")
    all_smiles_with_score1 = [s for s, score in ga_output.scoring_func_evals.items() if score == 1.0]
    num_atoms = [Chem.MolFromSmiles(s).GetNumAtoms() for s in all_smiles_with_score1]
    for s, n in sorted(zip(all_smiles_with_score1, num_atoms), key=lambda x: x[1], reverse=True):
        print(f"{n}\t\t\t{s}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("target_smiles", type=str)
    parser.add_argument("--generations", type=int, default=100)
    args = parser.parse_args()
    main(args.target_smiles, args.generations)
