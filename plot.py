from pathlib import Path
import json
import matplotlib.pyplot as plt


def plot_n_qubits(targets: list[str], data_directory: Path, figure_directory: Path) -> None:
    fig = plt.figure()
    ax = fig.add_subplot()

    for target in targets:
        data_file = data_directory / f"{target}.json"
        with data_file.open() as f:
            data = json.load(f)
            n_qubits_begin = data["n_qubits_begin"]
            n_qubits_end = data["n_qubits_end"]
            n_qubits = list(range(n_qubits_begin, n_qubits_end))
            durations = data["durations"]
            plt.plot(n_qubits, durations, label=target)

    ax.set_xlabel("n_qubits")
    ax.set_ylabel("Duration (ns)")
    ax.set_yscale("log")
    ax.legend()
    fig.gca().set_yscale("log")
    fig.gca().grid(linestyle="--")
    fig.savefig(f"{figure_directory}/n_qubits.png")


def main() -> None:
    targets = ["qulacs"]
    figure_directory = Path("figures")
    figure_directory.mkdir(exist_ok=True)

    plot_n_qubits(targets, Path("output"), figure_directory)


if __name__ == "__main__":
    main()
