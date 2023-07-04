from dataclasses import asdict, dataclass
from tqdm import tqdm
import click
import json
import os
import statistics
import subprocess


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
    # Directory at which benchmark code is located.
    # For example, if you test ./benchmarks/qulacs, then directory is "qulacs".
    directory: str
    n_qubits_begin: int
    n_qubits_end: int
    n_repeat: int

    def build_image(self) -> None:
        user_id = os.getuid()
        group_id = os.getgid()
        print("Building image")
        # Pass UID and GID to create user with same UID and GID as host user.
        subprocess.run(["docker", "build", "--build-arg", f"USER_ID={user_id}", "--build-arg", f"GROUP_ID={group_id}", "-t", f"qulacs2023_benchmarks/{self.directory}:latest", f"./benchmarks/{self.directory}/"], capture_output=True)

    def run_benchmark(self) -> BenchmarkResult:
        self.build_image()

        current_directory = os.getcwd()
        mount_config = f"type=bind,source={current_directory}/benchmarks/{self.directory},target=/benchmarks"
        image_tag = f"qulacs2023_benchmarks/{self.directory}:latest"

        print("Running benchmark program")
        build_result = subprocess.run(["docker", "run", "--rm", "-it", "--gpus", "all", "--mount", mount_config, image_tag, "/benchmarks/build.sh"], capture_output=True)
        if build_result.returncode != 0:
            raise RuntimeError(f"Failed to build a program in {self.directory}.")

        print("Running benchmark")
        means = []
        for n_qubits in tqdm(range(self.n_qubits_begin, self.n_qubits_end)):
            run_result = subprocess.run(["docker", "run", "--rm", "-it", "--gpus", "all", "--mount", mount_config, image_tag, "/benchmarks/main", f"{n_qubits}", f"{self.n_repeat}"], capture_output=True)
            if run_result.returncode != 0:
                raise RuntimeError(f"Failed to run {self.directory} image: {run_result.stderr}")

            with open(f"./benchmarks/{self.directory}/durations.txt", "r") as f:
                mean = BenchmarkCase.calculate_mean(f.read())
                means.append(mean)
        print("Finished benchmark")

        return BenchmarkResult(self.directory, self.n_qubits_begin, self.n_qubits_end, self.n_repeat, means)

    def calculate_mean(output: str) -> float:
        """Calculate mean of time from output of subprocess.run()"""
        durations = list(map(float, output.split()))
        return statistics.fmean(durations)


@click.command
@click.option("--directory", "-d", default=["qulacs"], multiple=True, help="Directory name of benchmark code")
@click.option("--n_qubits_begin", "-b", default=2, type=int, help="Number of qubits to start benchmark")
@click.option("--n_qubits_end", "-e", default=26, type=int, help="Number of qubits to end benchmark, exclusive")
@click.option("--n_repeat", "-r", default=10, type=int, help="Number of times to repeat benchmark")
def main(directory: list[str], n_qubits_begin: int, n_qubits_end: int, n_repeat: int):
    cases = [BenchmarkCase(d, n_qubits_begin, n_qubits_end, n_repeat) for d in directory]
    results = [case.run_benchmark() for case in cases]
    for result in results:
        result.save("output")


main()
