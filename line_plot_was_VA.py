import pandas as pd
from scipy.stats import wasserstein_distance
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from collections import Counter
import numpy as np


def wasserstein_trace(shares_df1, shares_df2, weights1, weights2, resolution=10_000):
    assert all(shares_df1.columns == shares_df2.columns)

    shares1 = shares_df1.sort_index(axis=1).to_numpy()
    shares2 = shares_df2.sort_index(axis=1).to_numpy()

    n_districts = len(shares1[0])

    assert len(shares1[0]) == len(shares2[0])
    state1 = np.zeros(n_districts)
    state2 = np.zeros(n_districts)
    xticks = []
    trace = []
    hist1 = [Counter() for _ in range(n_districts)]
    hist2 = [Counter() for _ in range(n_districts)]
    for step, (s1, s2, w1, w2) in enumerate(
        tqdm(zip(shares1, shares2, weights1, weights2), total=shares1.shape[0])
    ):
        # We assume 1-indexed districts.
        for dist, v in enumerate(s1):
            state1[dist] = v
        for dist, v in enumerate(s2):
            state2[dist] = v
        for k, v in enumerate(sorted(state1)):
            hist1[k][v] += w1
        for k, v in enumerate(sorted(state2)):
            hist2[k][v] += w2
        if step > 0 and step % resolution == 0:
            distance = 0
            for dist1, dist2 in zip(hist1, hist2):
                distance += wasserstein_distance(
                    list(dist1.keys()),
                    list(dist2.keys()),
                    list(dist1.values()),
                    list(dist2.values()),
                )
            xticks.append(step)
            trace.append(distance)
    return xticks, trace


def wasserstein_trace_v_full(
    shares_df, full_df, weights, weights_full, resolution=10_000
):
    assert all(shares_df.columns == full_df.columns)

    shares1 = shares_df.sort_index(axis=1).to_numpy()
    shares2 = full_df.sort_index(axis=1).to_numpy()

    n_districts = len(shares1[0])

    assert len(shares1[0]) == len(shares2[0])
    state1 = np.zeros(n_districts)
    state2 = np.zeros(n_districts)
    xticks = []
    trace = []
    hist1 = [Counter() for _ in range(n_districts)]
    hist2 = [Counter() for _ in range(n_districts)]

    for s2, w2 in zip(shares2, weights_full):
        for dist, v in enumerate(s2):
            state2[dist] = v
        for k, v in enumerate(sorted(state2)):
            hist2[k][v] += w2

    for step, (s1, w1) in enumerate(
        tqdm(zip(shares1, weights), total=shares1.shape[0])
    ):
        # We assume 1-indexed districts.
        for dist, v in enumerate(s1):
            state1[dist] = v
        for k, v in enumerate(sorted(state1)):
            hist1[k][v] += w1
        if step > 0 and step % resolution == 0:
            distance = 0
            for dist1, dist2 in zip(hist1, hist2):
                distance += wasserstein_distance(
                    list(dist1.keys()),
                    list(dist2.keys()),
                    list(dist1.values()),
                    list(dist2.values()),
                )
            xticks.append(step)
            trace.append(distance)
    return xticks, trace


if __name__ == "__main__":
    rev_df1 = pd.read_parquet(
        "./data/VA_RevRecom_steps_15000000000_rng_seed_278986_plan_CD_12_20241113_095352_tallies.parquet"
    )
    rev_df2 = pd.read_parquet(
        "./data/VA_RevRecom_steps_15000000000_rng_seed_278986_plan_CD_16_20241113_095352_tallies.parquet"
    )
    rev_df3 = pd.read_parquet(
        "./data/VA_RevRecom_steps_15000000000_rng_seed_278986_plan_rand_dist_eps0p01_20241113_095352_tallies.parquet"
    )
    forest_df = pd.read_parquet(
        "./data/VA_Forest_steps_10000000_rng_seed_278986_gamma_0.0_alpha_1.0_ndists_11_20241108_130331_tallies.parquet"
    )

    rev_df1_dem = rev_df1[rev_df1["sum_columns"] == "G16DPRS"].reset_index()
    rev_df1_rep = rev_df1[rev_df1["sum_columns"] == "G16RPRS"].reset_index()
    rev_df1_shares_total = rev_df1_dem[[f"district_{i}" for i in range(1, 12)]] / (
        rev_df1_dem[[f"district_{i}" for i in range(1, 12)]]
        + rev_df1_rep[[f"district_{i}" for i in range(1, 12)]]
    )
    rev_df1_shares_total.rename(
        columns={f"district_{i}": f"district_{i:02d}" for i in range(1, 12)},
        inplace=True,
    )
    rev_df1_shares_total.sort_index(axis=1, inplace=True)
    rev_df1_weights = rev_df1["n_reps"].to_numpy()

    rev_df2_dem = rev_df2[rev_df2["sum_columns"] == "G16DPRS"].reset_index()
    rev_df2_rep = rev_df2[rev_df2["sum_columns"] == "G16RPRS"].reset_index()
    rev_df2_shares_total = rev_df2_dem[[f"district_{i}" for i in range(1, 12)]] / (
        rev_df2_dem[[f"district_{i}" for i in range(1, 12)]]
        + rev_df2_rep[[f"district_{i}" for i in range(1, 12)]]
    )
    rev_df2_shares_total.rename(
        columns={f"district_{i}": f"district_{i:02d}" for i in range(1, 12)},
        inplace=True,
    )
    rev_df2_shares_total.sort_index(axis=1, inplace=True)
    rev_df2_weights = rev_df2["n_reps"].to_numpy()

    rev_df3_dem = rev_df3[rev_df3["sum_columns"] == "G16DPRS"].reset_index()
    rev_df3_rep = rev_df3[rev_df3["sum_columns"] == "G16RPRS"].reset_index()
    rev_df3_shares_total = rev_df3_dem[[f"district_{i}" for i in range(1, 12)]] / (
        rev_df3_dem[[f"district_{i}" for i in range(1, 12)]]
        + rev_df3_rep[[f"district_{i}" for i in range(1, 12)]]
    )
    rev_df3_shares_total.rename(
        columns={f"district_{i}": f"district_{i:02d}" for i in range(1, 12)},
        inplace=True,
    )
    rev_df3_shares_total.sort_index(axis=1, inplace=True)
    rev_df3_weights = rev_df3["n_reps"].to_numpy()

    n_accepted = 10_000_000
    n_items = 800

    was_12_ticks, was_12_distances = wasserstein_trace(
        shares_df1=rev_df1_shares_total.iloc[:n_accepted, :],
        shares_df2=rev_df2_shares_total.iloc[:n_accepted, :],
        weights1=rev_df1_dem.iloc[:n_accepted, :]["n_reps"],
        weights2=rev_df2_dem.iloc[:n_accepted, :]["n_reps"],
        resolution=n_accepted / n_items,
    )

    was_13_ticks, was_13_distances = wasserstein_trace(
        shares_df1=rev_df1_shares_total.iloc[:n_accepted, :],
        shares_df2=rev_df3_shares_total.iloc[:n_accepted, :],
        weights1=rev_df1_dem.iloc[:n_accepted, :]["n_reps"],
        weights2=rev_df3_dem.iloc[:n_accepted, :]["n_reps"],
        resolution=n_accepted / n_items,
    )

    was_23_ticks, was_23_distances = wasserstein_trace(
        shares_df1=rev_df2_shares_total.iloc[:n_accepted, :],
        shares_df2=rev_df3_shares_total.iloc[:n_accepted, :],
        weights1=rev_df2_dem.iloc[:n_accepted, :]["n_reps"],
        weights2=rev_df3_dem.iloc[:n_accepted, :]["n_reps"],
        resolution=n_accepted / n_items,
    )

    colors = [
        "#0099cd",
        "#00cd99",
        "#ffca5d",
        "#99cd00",
        "#cd0099",
        "#9900cd",
        "#8dd3c7",
        "#bebada",
        "#fb8072",
        "#80b1d3",
    ]

    _, ax = plt.subplots(figsize=(25, 10))

    sns.lineplot(
        x=was_12_ticks,
        y=was_12_distances,
        ax=ax,
        linewidth=3,
        color=colors[0],
        label="CD 16 vs CD 12",
    )
    sns.lineplot(
        x=was_13_ticks,
        y=was_13_distances,
        ax=ax,
        linewidth=3,
        color=colors[2],
        label="CD 16 vs Rand Dist",
    )
    sns.lineplot(
        x=was_23_ticks,
        y=was_23_distances,
        ax=ax,
        linewidth=3,
        color=colors[8],
        label="CD 12 vs Rand Dist",
    )

    plt.savefig(
        "./figures/VA/Wasserstein_distances_VA_comparison_Dem_Shares_All_Rev.png",
        bbox_inches="tight",
    )

    forest_df_dem = forest_df[forest_df["sum_columns"] == "G16DPRS"].reset_index()
    forest_df_rep = forest_df[forest_df["sum_columns"] == "G16RPRS"].reset_index()
    forest_df_shares_total = forest_df_dem[[f"district_{i}" for i in range(1, 12)]] / (
        forest_df_dem[[f"district_{i}" for i in range(1, 12)]]
        + forest_df_rep[[f"district_{i}" for i in range(1, 12)]]
    )
    forest_df_shares_total.rename(
        columns={f"district_{i}": f"district_{i:02d}" for i in range(1, 12)},
        inplace=True,
    )
    forest_df_shares_total.sort_index(axis=1, inplace=True)
    forest_df_weights = forest_df["n_reps"].to_numpy()

    n_accepted = 1_900_000
    n_items = 800

    was_13_ticks, was_13_distances = wasserstein_trace(
        shares_df1=rev_df1_shares_total.iloc[:n_accepted, :],
        shares_df2=rev_df3_shares_total.iloc[:n_accepted, :],
        weights1=rev_df1_dem.iloc[:n_accepted, :]["n_reps"],
        weights2=rev_df3_dem.iloc[:n_accepted, :]["n_reps"],
        resolution=n_accepted / n_items,
    )

    was_1f_ticks, was_1f_distances = wasserstein_trace(
        shares_df1=rev_df1_shares_total.iloc[:n_accepted, :],
        shares_df2=forest_df_shares_total.iloc[:n_accepted, :],
        weights1=rev_df1_dem.iloc[:n_accepted, :]["n_reps"],
        weights2=forest_df_dem.iloc[:n_accepted, :]["n_reps"],
        resolution=n_accepted / n_items,
    )

    was_3f_ticks, was_3f_distances = wasserstein_trace(
        shares_df1=rev_df3_shares_total.iloc[:n_accepted, :],
        shares_df2=forest_df_shares_total.iloc[:n_accepted, :],
        weights1=rev_df3_dem.iloc[:n_accepted, :]["n_reps"],
        weights2=forest_df_dem.iloc[:n_accepted, :]["n_reps"],
        resolution=n_accepted / n_items,
    )

    _, ax = plt.subplots(figsize=(25, 10))

    sns.lineplot(
        x=was_13_ticks,
        y=was_13_distances,
        ax=ax,
        linewidth=3,
        color=colors[2],
        label="CD 16 vs Rand Dist",
    )
    sns.lineplot(
        x=was_1f_ticks,
        y=was_1f_distances,
        ax=ax,
        linewidth=3,
        color=colors[1],
        label="CD 16 vs Forest",
    )
    sns.lineplot(
        x=was_3f_ticks,
        y=was_3f_distances,
        ax=ax,
        linewidth=3,
        color=colors[3],
        label="Rand Dist vs Forest",
    )

    plt.savefig(
        "./figures/VA/Wasserstein_distances_VA_comparison_Dem_Shares_v_Forest.png",
        bbox_inches="tight",
    )

    n_accepted_full = 10_000_00
    n_items_full = 800

    was_full_1f_ticks, was_full_1f_distances = wasserstein_trace_v_full(
        shares_df=rev_df1_shares_total.iloc[:n_accepted_full, :],
        full_df=forest_df_shares_total,
        weights=rev_df1_dem.iloc[:n_accepted_full, :]["n_reps"],
        weights_full=forest_df_dem["n_reps"],
        resolution=n_accepted_full / n_items_full,
    )

    was_full_2f_ticks, was_full_2f_distances = wasserstein_trace_v_full(
        shares_df=rev_df2_shares_total.iloc[:n_accepted_full, :],
        full_df=forest_df_shares_total,
        weights=rev_df2_dem.iloc[:n_accepted_full, :]["n_reps"],
        weights_full=forest_df_dem["n_reps"],
        resolution=n_accepted_full / n_items_full,
    )

    was_full_3f_ticks, was_full_3f_distances = wasserstein_trace_v_full(
        shares_df=rev_df3_shares_total.iloc[:n_accepted_full, :],
        full_df=forest_df_shares_total,
        weights=rev_df3_dem.iloc[:n_accepted_full, :]["n_reps"],
        weights_full=forest_df_dem["n_reps"],
        resolution=n_accepted_full / n_items_full,
    )

    _, ax = plt.subplots(figsize=(25, 10))

    sns.lineplot(
        x=was_full_1f_ticks,
        y=was_full_1f_distances,
        ax=ax,
        linewidth=3,
        color=colors[1],
        label="CD 16 vs Full Forest",
    )
    sns.lineplot(
        x=was_full_2f_ticks,
        y=was_full_2f_distances,
        ax=ax,
        linewidth=3,
        color=colors[0],
        label="CD 12 vs Full Forest",
    )
    sns.lineplot(
        x=was_full_3f_ticks,
        y=was_full_3f_distances,
        ax=ax,
        linewidth=3,
        color=colors[3],
        label="Rand Dist vs Full Forest",
    )

    plt.savefig(
        "./figures/VA/Wasserstein_distances_VA_comparison_Dem_Shares_v_Full_Forest.png",
        bbox_inches="tight",
    )
