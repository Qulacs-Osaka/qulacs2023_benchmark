from dataclasses import asdict, dataclass
from pathlib import Path
from tqdm import tqdm
import click
import json
import os
import statistics
import subprocess


@dataclass
class BenchmarkResult:
    target: str
    circuit_id: int
    n_qubits_begin: int
    n_qubits_end: int
    n_repeat: int
    durations: list[float]

    def save(self, output_directory: Path):
        data_file = output_directory / f"{self.target}_{self.circuit_id}.json"
        with data_file.open(mode="w") as f:
            d = asdict(self)
            json.dump(d, f, indent=4)


@dataclass
class BenchmarkCase:
    # Directory at which benchmark code is located.
    # For example, if you test ./benchmarks/qulacs, then directory is "qulacs".
    target: str
    circuit_id: int
    warmup: int
    n_qubits_begin: int
    n_qubits_end: int
    n_repeat: int

    def build_image(self) -> None:
        """Build docker image for benchmark"""
        user_id = os.getuid()
        group_id = os.getgid()
        print("Building image")
        # Pass UID and GID to create user with same UID and GID as host user.
        build_result = subprocess.run(["docker", "build", "--build-arg", f"USER_ID={user_id}", "--build-arg", f"GROUP_ID={group_id}", "-t", f"qulacs2023_benchmarks/{self.target}:latest", f"./benchmarks/{self.target}/"], capture_output=True)
        if build_result.returncode != 0:
            raise RuntimeError(f"Failed to build {self.target} image: {build_result.stderr}")

    def build_program(self, mount_config: str, image_tag:str) -> None:
        """Build benchmark program in the docker container"""
        print("Building benchmark program")
        build_result = subprocess.run(["docker", "run", "--rm", "-it", "--gpus", "all", "--mount", mount_config, image_tag, "/benchmarks/build.sh"], capture_output=True)
        if build_result.returncode != 0:
            raise RuntimeError(f"Failed to build a program in {self.target}.")

    def run_one_benchmark(self, n_qubits: int, mount_config: str, image_tag: str) -> float:
        """Run benchmark for one circuit and one number of qubits"""
        run_result = subprocess.run(["docker", "run", "--rm", "-it", "--gpus", "all", "--mount", mount_config, image_tag, "/benchmarks/main", f"{self.circuit_id}", f"{n_qubits}", f"{self.n_repeat}"], capture_output=True)
        if run_result.returncode != 0:
            raise RuntimeError(f"Failed to run {self.target} image: {run_result.stderr}")

        # Extract the benchmark data from not stdout, but a file because stdout is made dirty by output of CUDA image.
        durations_file = Path(f"./benchmarks/{self.target}/durations.txt")
        with durations_file.open() as f:
            mean = BenchmarkCase.calculate_mean(f.read())
            return mean

    def run_benchmark(self) -> BenchmarkResult:
        print(f"=== {self.target} ===")
        self.build_image()

        current_directory = os.getcwd()
        mount_config = f"type=bind,source={current_directory}/benchmarks/{self.target},target=/benchmarks"
        image_tag = f"qulacs2023_benchmarks/{self.target}:latest"

        self.build_program(mount_config, image_tag)

        for _ in range(self.warmup):
            # Fixed to 3 qubits for warmup. No special reason.
            self.run_one_benchmark(3, mount_config, image_tag)

        print(f"Running benchmark for {self.target}, circuit {self.circuit_id}")
        means = []
        for n_qubits in tqdm(range(self.n_qubits_begin, self.n_qubits_end)):
            mean = self.run_one_benchmark(n_qubits, mount_config, image_tag)
            means.append(mean)
        print("Finished benchmark")

        return BenchmarkResult(self.target, self.circuit_id, self.n_qubits_begin, self.n_qubits_end, self.n_repeat, means)

    def calculate_mean(output: str) -> float:
        """Calculate mean of time from output of subprocess.run()"""
        durations = list(map(float, output.split()))
        return statistics.fmean(durations)


@click.command
@click.option("--target", "-t", default=["qulacs"], multiple=True, help="Benchmark target. Specify directory name of benchmark code")
@click.option("--circuits", "-c", default=[0, 1, 2, 3, 4, 5], multiple=True, help="Circuit type. Specify circuit ID defined in README.md")
@click.option("--warmup", "-w", default=3, type=int, help="Number of times to discard benchmark results ahead of actual benchmark")
@click.option("--n-qubits-begin", "-b", default=2, type=int, help="Number of qubits to start benchmark")
@click.option("--n-qubits-end", "-e", default=26, type=int, help="Number of qubits to end benchmark, exclusive")
@click.option("--n-repeat", "-r", default=10, type=int, help="Number of times to repeat benchmark")
def main(target: list[str], circuits: list[int], warmup: int, n_qubits_begin: int, n_qubits_end: int, n_repeat: int):
    cases = [BenchmarkCase(t, c, warmup, n_qubits_begin, n_qubits_end, n_repeat) for t in target for c in circuits]
    results = [case.run_benchmark() for case in cases]
    for result in results:
        result.save(Path("output"))


if __name__ == "__main__":
    main()