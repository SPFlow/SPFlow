import os
import csv
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    base_path = os.path.join('results', 'mnistgen_ent')

    exp_names = [f"mnistgen_ent1", f"mnistgen_ent2"]
    dfs = []
    for exp_name in exp_names:
        results_path = os.path.join(base_path, f"results_{exp_name}")
        file_name = f"log_{exp_name}.csv"
        path = os.path.join(results_path, file_name)
        dfs.append(pd.read_csv(path))
    df1, df2 = dfs

    plt.figure()
    plt.plot(df1["epoch"], df1["nll_loss"], label="NLL Loss, no entropy loss", linestyle=":")
    plt.plot(df1["epoch"], df1["gmm_ent_lb"], label="GMM Entropy LB, no entropy loss", linestyle=":")
    plt.plot(df2["epoch"], df2["nll_loss"], label="NLL Loss, alpha=1.0", linestyle="-.")
    plt.plot(df2["epoch"], df2["gmm_ent_lb"], label="GMM Entropy LB, alpha=1.0", linestyle="-.")
    plt.legend()
    plt.show()
    # data_t = torch.tensor(data.values)
    print("done")
