"""Three warm repeats per configuration on independently selected GPUs."""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="testset")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configurations = [
        ("standard-all", ["--mode", "standard"]),
        ("triton-all", ["--mode", "text_stream", "--backend", "triton"]),
        (
            "torch-binary",
            ["--mode", "text_stream", "--backend", "torch", "--binary-only"],
        ),
        (
            "triton-binary",
            ["--mode", "text_stream", "--backend", "triton", "--binary-only"],
        ),
    ]

    def run(gpu, configuration):
        name, flags = configuration
        results = []
        for repeat in range(args.repeats):
            report = args.output_dir / f"{name}-{repeat}.json"
            command = [
                sys.executable,
                "benchmarks/benchmark_text_stream.py",
                "--video",
                args.video,
                "--frames",
                "64",
                "--warmup-pass",
                "--report",
                str(report),
                *flags,
            ]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=".")
            completed = subprocess.run(command, env=env, capture_output=True, text=True)
            if completed.returncode:
                raise RuntimeError(completed.stderr[-10000:])
            result = json.loads(report.read_text())
            results.append(
                {
                    k: result[k]
                    for k in (
                        "compute_fps",
                        "median_seconds",
                        "p95_seconds",
                        "peak_allocated",
                    )
                }
            )
            print(
                json.dumps({"configuration": name, "repeat": repeat, **results[-1]}),
                flush=True,
            )
        return name, results

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(run, gpu, config) for gpu, config in enumerate(configurations)
        ]
        results = dict(future.result() for future in futures)
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
