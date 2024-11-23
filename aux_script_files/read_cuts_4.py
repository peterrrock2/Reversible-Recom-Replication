import pandas as pd


if __name__ == "__main__":
    df = pd.read_parquet("./outputs/4x4_100k_cut_edges.parquet")

    df = df.groupby('cut_edges').sum()
    
    df["probability"] = 100*df["n_reps"] / df["n_reps"].sum()
    df = df[["probability"]].reset_index()
    print(df.to_string())
