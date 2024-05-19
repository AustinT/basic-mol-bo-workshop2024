from __future__ import annotations

import heapq
import logging
import random
import sys
from pprint import pformat
from typing import Callable

import gpytorch
import joblib
import numpy as np
import torch
from mol_ga import default_ga
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from trf23.tanimoto_gp import TanimotoKernelGP, batch_predict_mu_std_numpy

rdBase.DisableLog("rdApp.error")

FP_DIM = 2048
FP_RADIUS = 4
GP_MEAN = 0.00

stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(formatter)

acq_opt_logger = logging.getLogger("acq_opt_logger")
bo_loop_logger = logging.getLogger("bo_loop_logger")
bo_loop_logger.setLevel(logging.DEBUG)
bo_loop_logger.addHandler(stream_handler)


def smiles_to_fingerprint_arr(
    smiles_list: list[str],
) -> np.array:
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_DIM)
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = [mfpgen.GetCountFingerprintAsNumPy(m) for m in mols]
    return np.asarray(fps, dtype=float)


def get_gp_pred_on_smiles(
    smiles_list: list[str],
    model,
    device: torch.device,
    screen_batch_size: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    fps = smiles_to_fingerprint_arr(
        smiles_list,
    )
    return batch_predict_mu_std_numpy(model, fps, device=device, batch_size=screen_batch_size)


def run_tanimoto_gpbo(
    *,
    oracle: Callable[[list[str]], list[float]],
    smiles_bank: list[str],
    rng: random.Random,  # random number generator
    oracle_budget: int,
    num_start_samples: int = 10,
    max_bo_iter: int = 990,
    bo_batch_size: int = 1,
    ga_start_population_size: int = 10_000,
    ga_population_size: int = 10000,
    ga_max_generations: int = 5,
    ga_offspring_size: int = 200,
    max_heavy_atoms: int = 100,
) -> None:
    bo_loop_logger.info("Starting BO loop...")

    # Canonicalize all smiles and remove duplicates
    # (otherwise can could potentially cause bugs)
    bo_loop_logger.info("Canonicalizing all smiles")
    smiles_bank = list(set([Chem.CanonSmiles(s) for s in smiles_bank]))

    # Randomly choose initial smiles
    starting_population = rng.sample(smiles_bank, num_start_samples)
    starting_population_scores = oracle(starting_population)
    known_smiles_scores = {s: score for s, score in zip(starting_population, starting_population_scores)}

    # Run BO loop
    bo_iter = 0
    while len(known_smiles_scores) < oracle_budget:
        bo_iter += 1
        bo_loop_logger.info(f"Start BO iteration {bo_iter}")

        # Featurize known smiles
        smiles_train = list(known_smiles_scores.keys())
        scores_train = np.asarray([known_smiles_scores[s] for s in smiles_train])
        fp_train = smiles_to_fingerprint_arr(
            smiles_train,
        )

        # Make GP and set hyperparameters,
        # also turning off gradients because we won't fit the model
        torch.set_default_dtype(torch.float64)  # higher precision for GP
        gp_model = TanimotoKernelGP(
            train_x=torch.as_tensor(fp_train),
            train_y=torch.as_tensor(scores_train),
            kernel="T_MM",
            mean_obj=gpytorch.means.ConstantMean(),
        )
        gp_model.covar_module.raw_outputscale.requires_grad_(False)
        gp_model.mean_module.constant.requires_grad_(False)
        gp_model.likelihood.raw_noise.requires_grad_(False)
        gp_model.mean_module.constant.data.fill_(GP_MEAN)
        gp_model.covar_module.outputscale = 1.0
        gp_model.likelihood.noise = 1e-4  # little (but non-zero) noise, mostly for model misspecification
        gp_model.eval()

        # Potentially move to GPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gp_model = gp_model.cuda()
        else:
            device = torch.device("cpu")

        # Define acqusition function
        # We use UCB with random beta
        # chosen to be between 0.01 (pure exploitation)
        # and 1.0 (pure exploration, i.e. random points will have a higher value than incumbant best point).
        # For the last round, we use beta=0 (pure exploitation).
        if bo_iter <= max_bo_iter:
            ucb_beta = 10.0 ** rng.uniform(-2, 0)
            batch_size_this_iter = bo_batch_size
        else:
            ucb_beta = 0.0
            batch_size_this_iter = oracle_budget - len(known_smiles_scores)  # fill the budget
        bo_loop_logger.info(f"UCB beta: {ucb_beta:.3f}")

        # Pick starting population for acq opt GA
        ga_start_smiles = rng.choices(smiles_bank, k=ga_start_population_size) + list(known_smiles_scores.keys())
        ga_start_smiles = list(set(ga_start_smiles))  # remove duplicates

        # Optimize acquisition function
        bo_loop_logger.debug("Starting acquisition function optimization")

        def acq_fn(smiles_list):
            mu, std = get_gp_pred_on_smiles(smiles_list, gp_model, device)
            return (mu + ucb_beta * std).tolist()

        with joblib.Parallel(n_jobs=4) as parallel:
            acq_opt_output = default_ga(
                starting_population_smiles=ga_start_smiles,
                scoring_function=acq_fn,
                max_generations=ga_max_generations,
                offspring_size=ga_offspring_size,
                population_size=ga_population_size,
                rng=rng,
                parallel=parallel,
            )
        top_ga_smiles = sorted(acq_opt_output.scoring_func_evals.items(), key=lambda x: x[1], reverse=True)
        batch_candidate_smiles = iter([s for s, _ in top_ga_smiles])

        # Choose a batch of the top SMILES to evaluate which
        # 1) have not been measured before
        # 2) are unique
        # 3) not too many atoms
        eval_batch: list[str] = []
        while len(eval_batch) < batch_size_this_iter:
            try:
                s = next(batch_candidate_smiles)
                if s not in known_smiles_scores and s not in eval_batch:
                    mol = Chem.MolFromSmiles(s)
                    if mol is not None and mol.GetNumHeavyAtoms() <= max_heavy_atoms:
                        eval_batch.append(s)
                    del mol
            except StopIteration:
                break

        # Log info about the batch
        mu_batch, std_batch = get_gp_pred_on_smiles(eval_batch, gp_model, device)
        eval_batch_acq_values = [acq_opt_output.scoring_func_evals[s] for s in eval_batch]
        bo_loop_logger.debug(f"Eval batch SMILES: {pformat(eval_batch)}")
        bo_loop_logger.debug(f"Eval batch acq values: {eval_batch_acq_values}")
        bo_loop_logger.debug(f"Eval batch mu: {mu_batch.tolist()}")
        bo_loop_logger.debug(f"Eval batch std: {std_batch.tolist()}")

        # Score the batch with the oracle
        eval_batch_scores = oracle(eval_batch)
        bo_loop_logger.debug(f"Eval batch scores: {eval_batch_scores}")
        known_smiles_scores.update({s: score for s, score in zip(eval_batch, eval_batch_scores)})

        # Final message
        bo_loop_logger.info(
            f"End BO iteration {bo_iter}. Top scores so far:\n{pformat(heapq.nlargest(5, known_smiles_scores.values()))}"
        )

        # Free up GPU memory for next iteration by deleting the model
        del gp_model
        torch.cuda.empty_cache()

    bo_loop_logger.info("Finished BO loop.")
