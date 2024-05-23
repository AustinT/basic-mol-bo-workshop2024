import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles1", type=str)
    parser.add_argument("smiles2", type=str)
    parser.add_argument("output_file", type=str)
    args = parser.parse_args()

    # Draw the two molecules wtih rdkit
    from rdkit import Chem
    from rdkit.Chem import Draw

    img = Draw.MolsToGridImage(
        [Chem.MolFromSmiles(args.smiles1), Chem.MolFromSmiles(args.smiles2)],
        molsPerRow=2,
        subImgSize=(1000, 1000),
    )
    img.save(args.output_file)
