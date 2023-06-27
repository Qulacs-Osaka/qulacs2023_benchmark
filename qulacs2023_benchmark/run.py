from dataclasses import asdict, dataclass
import json
import statistics
import subprocess

import click

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
            json.dump(d, f, indent=4)


@dataclass
class BenchmarkCase:
    # Directory in which benchmark code is located
    directory: str
    n_qubits_begin: int
    n_qubits_end: int
    n_repeat: int

    def build_image(self) -> None:
        subprocess.run(["docker", "build", "-t", f"qulacs2023_benchmarks/{self.directory}:latest", f"./benchmarks/{self.directory}/"])

    def run_benchmark(self) -> BenchmarkResult:
        means = []
        self.build_image()
        for n_qubits in range(self.n_qubits_begin, self.n_qubits_end + 1):
            mount_config = f"type=bind,source=$(pwd)/benchmarks/{self.directory},target=/benchmarks"
            image_tag = f"qulacs2023_benchmarks/{self.directory}:latest"
            subprocess.run(["docker", "run", "--rm", "-it", "--gpus", "all", "--mount", mount_config, image_tag, "/benchmarks/build.sh"], capture_output=True)
            output = subprocess.run(["docker", "run", "--rm", "-it", "--gpus", "all", "--mount", mount_config, image_tag, "/benchmarks/main", f"{n_qubits}", f"{self.n_repeat}"], capture_output=True)
            mean = BenchmarkCase.calculate_mean(output.stdout)
            means.append(mean)
        return BenchmarkResult(self.directory, self.n_qubits_begin, self.n_qubits_end, self.n_repeat, means)

    def calculate_mean(output: bytes) -> float:
        """Calculate mean of time from output of subprocess.run()"""
        output_str = output.decode("utf-8").strip()
        durations = list(map(float, output_str.split()))
        return statistics.fmean(durations)


@click.command
@click.option("--directory", "-d", default=["qulacs"], multiple=True)
@click.option("--n_qubits_begin", "-b", default=2, type=int, help="Number of qubits to start benchmark")
@click.option("--n_qubits_end", "-e", default=25, type=int, help="Number of qubits to end benchmark, inclusive")
@click.option("--n_repeat", "-r", default=10, type=int, help="Number of times to repeat benchmark")
def main(directory: list[str], n_qubits_begin: int, n_qubits_end: int, n_repeat: int):
    cases = [BenchmarkCase(d, n_qubits_begin, n_qubits_end, n_repeat) for d in directory]
    results = [case.run_benchmark() for case in cases]
    for result in results:
        result.save("output")


main()
