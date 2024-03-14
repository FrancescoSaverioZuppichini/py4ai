from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import scienceplots


def plot_data(
    input_file,
    aggregation_keys,
    aggregation_name,
    group_by_name,
    x_axis,
    y_axis,
    x_axis_lim,
    y_axis_lim,
    name="",
    output_file=None,
    output_dir=None,
    sort_by=None, 
):
    plt.style.use(["science"])
    df = pd.read_json(input_file, lines=True)
    if aggregation_name == "mean":
        aggregated_df = df.groupby(aggregation_keys).mean().reset_index()
    elif aggregation_name == "sum":
        aggregated_df = df.groupby(aggregation_keys).sum().reset_index()
    elif aggregation_name == "count":
        aggregated_df = df.groupby(aggregation_keys).count().reset_index()
    if sort_by:
        aggregated_df = aggregated_df.sort_values(by=sort_by)
    print(aggregated_df)
    plt.figure(figsize=(10, 6), dpi=300)
    for key, grp in aggregated_df.groupby(group_by_name):
        print(group_by_name)
        plt.plot(
            grp[x_axis],
            grp[y_axis],
            label=f"{','.join([el for el in group_by_name])} {key}",
            marker="o",
        )
    plt.xlabel(x_axis.capitalize())
    plt.ylabel(y_axis.capitalize())
    if x_axis_lim:
        plt.xlim(
            left=x_axis_lim[0], right=(x_axis_lim[1] if len(x_axis_lim) > 1 else None)
        )
    if y_axis_lim:
        plt.ylim(
            bottom=y_axis_lim[0], top=(y_axis_lim[1] if len(y_axis_lim) > 1 else None)
        )
    plt.title(
        f"Plot of {y_axis.capitalize()} vs. {x_axis.capitalize()} for {','.join([el.capitalize() for el in group_by_name])}"
    )
    plt.legend()
    if not output_file:
        output_file = f"{name}_{'-'.join([el for el in group_by_name])}_{x_axis}_vs_{y_axis}.png"
    if output_dir:
        output_file = str(output_dir / output_file)
    plt.savefig(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from JSONL data.")
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input JSONL file"
    )
    parser.add_argument(
        "--aggregation_keys",
        nargs="+",
        required=True,
        help="List of keys to aggregate on",
    )
    parser.add_argument(
        "--aggregation_name",
        type=str,
        choices=["mean", "sum", "count"],
        required=False,
        default="sum",
        help="Aggregation function",
    )
    parser.add_argument(
        "--group_by_name", nargs="+", required=True, help="Key to group by for plotting"
    )
    parser.add_argument(
        "--x_axis", type=str, required=True, help="Variable for the x-axis"
    )
    parser.add_argument(
        "--y_axis", type=str, required=True, help="Variable for the y-axis"
    )
    parser.add_argument(
        "--x_axis_lim", nargs="+", type=float, help="Optional x-axis limits (min [max])"
    )
    parser.add_argument(
        "--y_axis_lim", nargs="+", type=float, help="Optional y-axis limits (min [max])"
    )
    parser.add_argument("--name", type=str, help="Optional name of the plot")
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Optional output dir for the output plots for the plot",
    )
    parser.add_argument(
        "--output_file", type=str, help="Optional output file name for the plot"
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        help="Optional column name to sort the data"
    )

    args = parser.parse_args()
    if args.output_dir:
        args.output_dir.mkdir(exist_ok=True, parents=True)

    plot_data(
        args.input_file,
        args.aggregation_keys,
        args.aggregation_name,
        args.group_by_name,
        args.x_axis,
        args.y_axis,
        args.x_axis_lim,
        args.y_axis_lim,
        args.name,
        args.output_file,
        args.output_dir,
        args.sort_by
    )
