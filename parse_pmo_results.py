import argparse
import json
import statistics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_files", nargs="+", type=str)
    parser.add_argument("--metric", type=str, default="top10-AUC")
    args = parser.parse_args()

    # Load the results from the files, assuming file format
    # {method}__{objective}__{trial}.json
    results = {}
    for results_file in args.results_files:
        with open(results_file, "r") as f:
            d = json.load(f)
        method, objective, trial = results_file.split("__")
        trial = int(trial.split(".")[0])
        results.setdefault(method, {}).setdefault(objective, []).append(d[args.metric])

    # Print the results
    for method, method_results in results.items():
        print(method)
        mean_cumsum = 0.0
        for objective, objective_results in sorted(method_results.items()):
            mean = statistics.mean(objective_results)
            std = statistics.stdev(objective_results)
            print(f"{objective:<25s} {mean:.3f} ± {std:.3f} ({len(objective_results)} trials)")
            mean_cumsum += mean
        print(f"{'Total':<25s} {mean_cumsum:.3f}")
        print()
