import argparse
import linktransformer as lt
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--df_path",
        type=str,
        default="data/large_dataset.csv",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="paraphrase-multilingual-mpnet-base-v2",
    )
    parser.add_argument(
        "--col_name",
        type=str,
        nargs="+",
        default="building_name",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )
    return parser.parse_args()


def dedup_rows(df_input_path, model_name, col_name, threshold):
    df = pd.read_csv(df_input_path)
    df_col = pd.DataFrame()
    df_col[col_name] = df[col_name]
    return lt.dedup_rows(
        df=df_col,
        model=model_name,
        on=col_name,
        cluster_type="agglomerative",
        cluster_params={"threshold": threshold},
    )


def cluster_rows(df_input_path, model_name, col_name, threshold):
    df = pd.read_csv(df_input_path)
    df_col = pd.DataFrame()
    df_col[col_name] = df[col_name].dropna()
    df_col["location_info"] = df["location_info"]
    return lt.cluster_rows(
        df=df_col,
        model=model_name,
        on=col_name,
        cluster_type="agglomerative",
        cluster_params={"threshold": threshold},
    )


if __name__ == "__main__":
    args = get_args()
    threshold = args.threshold
    col_name = args.col_name
    col_name = ["building_name", "country"]
    a = cluster_rows(
        df_input_path=args.df_path,
        model_name=args.model_name,
        col_name=col_name,
        threshold=threshold,
    )
    col_name = col_name if isinstance(col_name, list) else [col_name]
    b = a[col_name + ["cluster", "location_info"]]
    b[f"clustered_{col_name}"] = b.groupby("cluster")[col_name].transform(
        lambda x: x.iloc[0]
    )
    print(
        "Saving to data/{}_clusters_0{}_threshold.csv".format(col_name, threshold * 10)
    )
    b.to_csv(
        f"data/{col_name}_clusters_0{int(threshold * 10)}_threshold.csv", index=False
    )

    pass
