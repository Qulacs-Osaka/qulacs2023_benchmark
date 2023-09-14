import json
import os
import statistics
import subprocess
import typing
from dataclasses import asdict, dataclass
from pathlib import Path

import click
from tqdm import tqdm

AggregateBy = typing.Literal["average", "median"]


@dataclass
class BenchmarkResult:
    target: str
    circuit_id: int
    n_qubits_begin: int
    n_qubits_end: int
    n_repeat: int
    durations: list[float]

    def save(self, output_directory: Path) -> None:
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
    aggregate_by: AggregateBy

    def run_one_benchmark(self, n_qubits: int, mount_config: str, image_tag: str) -> float:
        """Run benchmark for one circuit and one number of qubits"""
        run_result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-it",
                "--gpus",
                "all",
                "--mount",
                mount_config,
                image_tag,
                "/benchmarks/main",
                f"{self.circuit_id}",
                f"{n_qubits}",
                f"{self.n_repeat}",
            ],
            capture_output=True,
        )
        if run_result.returncode != 0:
            raise RuntimeError(f"Failed to run {self.target} image: {run_result.stderr.decode()}")

        # Extract the benchmark data from not stdout, but a file because stdout is made dirty by output of CUDA image.
        durations_file = Path(f"./benchmarks/{self.target}/durations.txt")
        with durations_file.open() as f:
            mean = BenchmarkCase.aggregate(f.read(), self.aggregate_by)
            return mean

    def run_benchmark(self) -> BenchmarkResult:
        print(f"=== {self.target} ===")

        mount_config, image_tag = render_docker_config(self.target)

        for _ in range(self.warmup):
            # Fixed to 3 qubits for warmup. No special reason.
            self.run_one_benchmark(3, mount_config, image_tag)

        print(f"Running benchmark for {self.target}, circuit {self.circuit_id}")
        means = []
        for n_qubits in tqdm(range(self.n_qubits_begin, self.n_qubits_end)):
            mean = self.run_one_benchmark(n_qubits, mount_config, image_tag)
            means.append(mean)
        print("Finished benchmark")

        return BenchmarkResult(
            self.target, self.circuit_id, self.n_qubits_begin, self.n_qubits_end, self.n_repeat, means
        )

    @staticmethod
    def aggregate(output: str, aggregate_by: AggregateBy) -> float:
        """Aggregate durations from output of subprocess.run()"""
        durations = list(map(float, output.split()))
        if aggregate_by == "average":
            return statistics.fmean(durations)
        elif aggregate_by == "median":
            return statistics.median(durations)
        else:
            raise ValueError(f"Unknown aggregate_by: {aggregate_by}")


def build_image(target: str) -> None:
    """Build docker image for benchmark"""
    user_id = os.getuid()
    group_id = os.getgid()
    print(f"Building image for {target}")
    # Pass UID and GID to create user with same UID and GID as host user.
    build_result = subprocess.run(
        [
            "docker",
            "build",
            "--build-arg",
            f"USER_ID={user_id}",
            "--build-arg",
            f"GROUP_ID={group_id}",
            "-t",
            f"qulacs2023_benchmarks/{target}:latest",
            f"./benchmarks/{target}/",
        ],
        capture_output=True,
    )
    if build_result.returncode != 0:
        raise RuntimeError(f"Failed to build {target} image: {build_result.stderr.decode()}")


def build_program(target: str, mount_config: str, image_tag: str) -> None:
    """Build benchmark program in the docker container"""
    print(f"Building benchmark program for {target}")
    build_result = subprocess.run(
        ["docker", "run", "--rm", "-it", "--gpus", "all", "--mount", mount_config, image_tag, "/benchmarks/build.sh"],
        capture_output=True,
    )
    if build_result.returncode != 0:
        raise RuntimeError(f"Failed to build a program in {target}.")


def render_docker_config(target: str) -> tuple[str, str]:
    """Render mount config and image tag for benchmark target"""
    current_directory = os.getcwd()
    mount_config = f"type=bind,source={current_directory}/benchmarks/{target},target=/benchmarks"
    image_tag = f"qulacs2023_benchmarks/{target}:latest"
    return mount_config, image_tag


@click.command
@click.option(
    "--target",
    "-t",
    default=["qulacs_now", "qulacs_cpu" "kokkos", "sycl"],
    multiple=True,
    help="Benchmark target. Specify directory name of benchmark code",
)
@click.option(
    "--circuits",
    "-c",
    default=[0, 1, 2, 3, 4, 5],
    multiple=True,
    help="Circuit type. Specify circuit ID defined in README.md",
)
@click.option(
    "--warmup", "-w", default=3, type=int, help="Number of times to discard benchmark results ahead of actual benchmark"
)
@click.option("--n-qubits-begin", "-b", default=3, type=int, help="Number of qubits to start benchmark")
@click.option("--n-qubits-end", "-e", default=26, type=int, help="Number of qubits to end benchmark, exclusive")
@click.option("--n-repeat", "-r", default=10, type=int, help="Number of times to repeat benchmark")
@click.option("--aggregate", "-a", default="average", type=click.Choice(["average", "median"]))
def main(
    target: list[str],
    circuits: list[int],
    warmup: int,
    n_qubits_begin: int,
    n_qubits_end: int,
    n_repeat: int,
    aggregate: AggregateBy,
) -> None:
    for t in target:
        mount_config, image_tag = render_docker_config(t)
        build_image(t)
        build_program(t, mount_config, image_tag)

    cases = [
        BenchmarkCase(t, c, warmup, n_qubits_begin, n_qubits_end, n_repeat, aggregate) for t in target for c in circuits
    ]
    results = [case.run_benchmark() for case in cases]
    for result in results:
        result.save(Path("output"))


if __name__ == "__main__":
    main()
