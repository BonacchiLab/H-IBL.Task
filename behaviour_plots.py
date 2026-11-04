'''This script reads CSV files from a folder or a single CSV file and generates plots for psychometric function 
and reaction time performance between blocks. These plots are useful for fitting the data to the similar behavioral data
seen in the original IBL task.'''


import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Psychometric function for contrast x behavior fitting, based on a logistic function.
def psychometric_function(x, bias, slope):
    return 100 / (1 + np.exp(-slope * (x - bias)))


def get_csv_paths(default_data_dir):
    """Ask for a CSV file or folder and return the CSV paths to process."""
    path_input = input(
        "Enter the path to a CSV file or folder "
        f"[default: {default_data_dir}]: "
    ).strip().strip('"')
    target_path = os.path.abspath(os.path.expanduser(path_input or default_data_dir))

    if os.path.isfile(target_path):
        if not target_path.lower().endswith(".csv"):
            raise ValueError(f"The selected file is not a CSV: {target_path}")
        csv_paths = [target_path]
    elif os.path.isdir(target_path):
        csv_paths = [
            os.path.join(folder, filename)
            for folder, _, filenames in os.walk(target_path)
            for filename in filenames
            if filename.lower().endswith(".csv")
        ]
        if not csv_paths:
            raise FileNotFoundError(f"No CSV files were found in {target_path}")
    else:
        raise FileNotFoundError(f"The file or folder was not found: {target_path}")

    print(f"Found {len(csv_paths)} CSV file(s).")
    return csv_paths


def load_data(csv_paths):
    """Read all CSV files and combine them into one DataFrame."""
    dataframes = []
    for csv_path in csv_paths:
        try:
            dataframes.append(pd.read_csv(csv_path, sep=None, engine="python"))
        except Exception as error:
            print(f"Error reading file {csv_path}: {error}")

    print(f"Successfully read {len(dataframes)} out of {len(csv_paths)} CSV files.")
    if not dataframes:
        raise ValueError("No valid data could be extracted from the CSV files found.")

    data = pd.concat(dataframes, ignore_index=True)
    data.columns = data.columns.str.strip()
    return data


def prepare_data(data):
    """Create the contrast and response columns used by the plots."""
    if "position" not in data.columns or "grating" not in data.columns:
        raise KeyError("The 'position' or 'grating' columns were not found in the file.")

    data["sign"] = data["position"].apply(lambda value: -1 if "-" in str(value) else 1)
    data["contrast"] = data["sign"] * data["grating"] * 100

    response_column = "keySide.keys"
    if response_column not in data.columns:
        raise KeyError(f"The response column '{response_column}' was not found in your CSV file.")

    data["choice_binary"] = data[response_column].apply(
        lambda value: 1
        if str(value).strip().lower() == "l"
        else (0 if str(value).strip().lower() == "s" else np.nan)
    )
    return data.dropna(subset=["choice_binary"])

# Main function to find the contrasts that better fit the psychometric.
def plot_psychometric(data):
    """Plot rightward-choice rate and the fitted psychometric curve."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    main_color = "#7E1010"
    x_smooth = np.linspace(-100, 100, 400)
    data = data.loc[~data["contrast"].abs().round(8).isin({2.5, 8})] # abs()
    stats = data.groupby("contrast")["choice_binary"].agg(["mean", "sem"]).reset_index()
    stats[["mean", "sem"]] *= 100
    x_data = stats["contrast"].values
    y_data = stats["mean"].values
    y_err = stats["sem"].values

    try:
        popt, _ = curve_fit(
            psychometric_function, x_data, y_data, p0=[0, 0.1], maxfev=5000
        )
        ax.plot(
            x_smooth,
            psychometric_function(x_smooth, *popt),
            color=main_color,
            lw=2.5,
            label="Psychometric fit",
        )
        print("\nFit completed successfully:")
        print(f" -> Bias (target point of indifference): {popt[0]:.2f}%")
        print(f" -> Slope (subject sensitivity): {popt[1]:.4f}")
        print(f" -> Mean reaction time: {data['keySide.rt'].mean() * 1000:.2f} ms")
    except Exception as error:
        print(f"\nCould not fit the logistic curve: {error}")

    ax.errorbar(
        x_data,
        y_data,
        yerr=y_err,
        fmt="o",
        color=main_color,
        mec="w",
        mew=0.5,
        ms=7,
        label="Subject data",
    )
    ax.set_xlabel(r"$\Delta$ Contrast (%)", fontsize=11)
    ax.set_ylabel("Rightward choices (%)", fontsize=11)
    ax.set_ylim(-5, 105)
    ax.set_xlim(-105, 105)
    x_ticks = [-100, -4, -2, -1, 0, 1, 2, 4, 100]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(tick) for tick in x_ticks], rotation=45)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Psychometric function")
    fig.tight_layout()


# Main function to find the contrasts that better fit the psychometric. 
# HAS ALL CONTRASTS
def plot_psychometric_all_contrasts(data):
    """Plot rightward-choice rate and the fitted psychometric curve."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    main_color = "#7E1010"
    x_smooth = np.linspace(-100, 100, 400)
    stats = data.groupby("contrast")["choice_binary"].agg(["mean", "sem"]).reset_index()
    stats[["mean", "sem"]] *= 100
    x_data = stats["contrast"].values
    y_data = stats["mean"].values
    y_err = stats["sem"].values

    try:
        popt, _ = curve_fit(
            psychometric_function, x_data, y_data, p0=[0, 0.1], maxfev=5000
        )
        ax.plot(
            x_smooth,
            psychometric_function(x_smooth, *popt),
            color=main_color,
            lw=2.5,
            label="Psychometric fit",
        )
        print("\nFit completed successfully:")
        print(f" -> Bias (target point of indifference): {popt[0]:.2f}%")
        print(f" -> Slope (subject sensitivity): {popt[1]:.4f}")
        print(f" -> Mean reaction time: {data['keySide.rt'].mean() * 1000:.2f} ms")
    except Exception as error:
        print(f"\nCould not fit the logistic curve: {error}")

    ax.errorbar(
        x_data,
        y_data,
        yerr=y_err,
        fmt="o",
        color=main_color,
        mec="w",
        mew=0.5,
        ms=7,
        label="Subject data",
    )
    ax.set_xlabel(r"$\Delta$ Contrast (%)", fontsize=11)
    ax.set_ylabel("Rightward choices (%)", fontsize=11)
    ax.set_ylim(-5, 105)
    ax.set_xlim(-105, 105)
    x_ticks = [-100,- 8, -4, -2.5, -2, -1, 0, 1, 2, 2.5, 4, 8, 100]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(tick) for tick in x_ticks], rotation=45)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Psychometric function")
    fig.tight_layout()




""" OTHER PLOTS"""
def plot_performance_between_blocks(data):
    """Plot reaction time by block."""
    block_column = "Pretest1Block.thisN"
    if block_column not in data.columns or data[block_column].nunique() < 2:
        print("Skipping block plot: at least two blocks are required.")
        return

    if "keySide.rt" not in data.columns:
        print("Skipping block plot: the 'keySide.rt' column was not found.")
        return

    valid = data[[block_column, "keySide.rt"]].copy()
    valid["keySide.rt"] = pd.to_numeric(valid["keySide.rt"], errors="coerce")
    valid = valid[valid["keySide.rt"] > 0]
    block_stats = valid.groupby(block_column)["keySide.rt"].mean() * 1000
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(block_stats.index, block_stats.values, marker="o", color="#7E1010")
    ax.set_xlabel("Block")
    ax.set_ylabel("Reaction time (ms)")
    ax.set_ylim(bottom=0)
    ax.set_title("Reaction time between blocks")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()


def main():
    start_time = time.perf_counter()
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    csv_paths = get_csv_paths(data_dir)
    data = prepare_data(load_data(csv_paths))

    plot_psychometric(data)
    plot_psychometric_all_contrasts(data)
    plot_performance_between_blocks(data)
    plt.show()

    end_time = time.perf_counter()
    print(f"Processing time: {end_time - start_time:.4f} seconds")





if __name__ == "__main__":
    main()