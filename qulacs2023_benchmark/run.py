from dataclasses import asdict, dataclass
import json
import statistics
import subprocess

import click

@dataclass
class BenchmarkCase:
    # Directory in which benchmark code is located
    directory: str
    n_qubits_begin: int
    n_qubits_end: int
    n_repeat: int


@dataclass
class BenchmarkResult:
    directory: str
    n_qubits_begin: int
    n_qubits_end: int
    n_repeat: int
    durations: list[float]

    def save(self, output_directory: str):
        with open(f"{output_directory}/{self.directory}.json", "w") as f:
            d = asdict(self)
            json.dump(d, f)


def calculate_mean(output: bytes) -> float:
    """Calculate mean of time from output of subprocess.run()"""
    output_str = output.decode("utf-8").strip()
    durations = list(map(float, output_str.split()))
    return statistics.fmean(durations)


def run_benchmark(case: BenchmarkCase):
    means = []
    for n_qubits in range(case.n_qubits_begin, case.n_qubits_end + 1):
        subprocess.run([f"./benchmark/{case.directory}/build.sh"], stdout=subprocess.DEVNULL)
        output = subprocess.run([f"./benchmark/{case.directory}/benchmark", str(n_qubits), str(case.n_repeat)], capture_output=True)
        mean = calculate_mean(output.stdout)
        means.append(mean)
    return BenchmarkResult(case.directory, case.n_qubits_begin, case.n_qubits_end, case.n_repeat, means)


@click.command
@click.option("--directory", "-d", default=["qulacs"], multiple=True)
@click.option("--n_qubits_begin", "-b", default=2)
@click.option("--n_qubits_end", "-e", default=25)
@click.option("--n_repeat", "-r", default=10)
def main(directory: list[str], n_qubits_begin: int, n_qubits_end: int, n_repeat: int):
    cases = [BenchmarkCase(d, n_qubits_begin, n_qubits_end, n_repeat) for d in directory]
    results = [run_benchmark(case) for case in cases]
    for result in results:
        result.save("results")

