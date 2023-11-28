import json
from pathlib import Path

import matplotlib.pyplot as plt


# For each circuit, plot n_qubits and the duration of execution time for all targets.
def plot_n_qubits(targets: list[str], circuit_id: int, data_directory: Path, figure_directory: Path) -> None:
    fig = plt.figure()
    ax = fig.add_subplot()

    for target in targets:
        data_file = data_directory / f"{target}_{circuit_id}.json"
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
    fig.savefig(f"{figure_directory}/n_qubits_{circuit_id}.png")


def main() -> None:
    #targets = [entry.name for entry in Path("benchmarks").iterdir() if entry.is_dir()]
    #targets = ["qulacs_cpu_single","qulacs_cpu_multi","kokkos_cpu_single","kokkos_cpu_multi"]
    targets = ["kokkos", "kokkos_new", "qulacs_gpu"]
    figure_directory = Path("figures")
    figure_directory.mkdir(exist_ok=True)

    #for circuit_id in range(6):
        #plot_n_qubits(targets, circuit_id, Path("output"), figure_directory)
    plot_n_qubits(targets, 3, Path("output"), figure_directory)
    plot_n_qubits(targets, 4, Path("output"), figure_directory)


if __name__ == "__main__":
    main()
