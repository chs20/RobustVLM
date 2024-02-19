import os

import pandas as pd
import argparse


def main(input_file):
    # Read the original CSV
    df = pd.read_csv(input_file)
    df['acc1'] = (df['acc1'] * 100).round(2)
    df['dataset'] = df['dataset'].str.replace(r'^(wds/vtab/|wds/)', '', regex=True)

    # Define the columns to pivot
    columns_to_pivot = ["dataset"]

    # Define the columns that will become the index of the new table
    index_columns = ["model", "pretrained", "attack", "eps", "iterations_adv"]

    # Pivot the DataFrame to the desired format
    df_pivot = df.pivot_table(values="acc1", index=index_columns, columns=columns_to_pivot).reset_index()
    del df

    # Save the pivoted DataFrame as a new CSV
    output_file = "pivoted.csv"
    df_pivot.to_csv(output_file, index=False)
    print(df_pivot, "\n")
    print(df_pivot.to_csv(index=False))
    print(f"Pivoted CSV saved as {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pivot a CSV file.")
    parser.add_argument("input_file", type=str, default=None, help="The input CSV file to be pivoted.")
    args = parser.parse_args()

    if not args.input_file:
        input_file = input("enter input file: ")
        # input_file = os.path.join("out", input_file)
    else:
        input_file = args.input_file

    main(input_file)
