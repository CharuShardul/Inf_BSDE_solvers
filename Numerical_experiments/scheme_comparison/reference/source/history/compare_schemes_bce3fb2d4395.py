"""CPU accuracy/time comparison using the three existing Brownian solvers.

Run: python compare_schemes.py --run
Replot cached results: python compare_schemes.py
Short pilot: python compare_schemes.py --pilot

Only library/runtime warm-up is excluded. Each independent solve includes model
construction, compilation, samples, targets, and optimization. Epochs/blocks are
atomic checkpoints: an update finishing after its budget is rolled back.
"""

import argparse
import contextlib
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Numerical_experiments" / "scheme_comparison"
PLOT = ROOT / "plots_article" / "scheme_comparison.pdf"
METHODS = ("grid", "picard", "direct")
CONFIGS = {
    "grid": [dict(Ntilde=n, M=m, R=6, r=0)
             for n, m in [(10, 2000), (20, 10000), (40, 40000)]],
    "picard": [dict(M=m, batch_size=1000, epochs=200)
               for m in [10000, 30000, 60000]],
    "direct": [dict(M=m, batch_size=512, update_frequency=100)
               for m in [300, 1000, 3000]],
}
VALIDATION_SEED = 24001
TEST_SEED = 24002
VALIDATION_SIZE = 10000
TEST_SIZE = 100000
VERSION = 1


def configure_cpu():
    """Called before numerical-library imports, including in child processes."""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS",
                "TF_NUM_INTEROP_THREADS"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "bsde_mpl_cache")
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    logging.basicConfig(handlers=[logging.NullHandler()])
    logging.disable(logging.CRITICAL)


def relative_errors(prediction, points):
    import numpy as np
    truth_u = np.mean(np.arctan(points), axis=-1)
    truth_ub = 1 / (points.shape[1] * (1 + points**2))
    err_u = np.sqrt(np.mean((prediction[:, 0] - truth_u)**2)
                    / np.mean(truth_u**2))
    err_ub = np.sqrt(np.mean(np.sum((prediction[:, 1:] - truth_ub)**2, axis=1))
                     / np.mean(np.sum(truth_ub**2, axis=1)))
    return {"relative_u": float(err_u), "relative_ub": float(err_ub),
            "score": float(max(err_u, err_ub))}


def choose_config(records, method, budget):
    """Selection only accepts tuning records; final test errors cannot tune it."""
    import numpy as np
    candidates = []
    for index in range(len(CONFIGS[method])):
        rows = [r for r in records if r["stage"] == "tune"
                and r["method"] == method and r["budget"] == budget
                and r["config_index"] == index]
        if not rows:
            raise ValueError("Missing tuning records")
        scores = [r["validation"]["score"] for r in rows]
        candidates.append(float(np.mean(scores)))
    return int(np.argmin(candidates)), candidates


def initialize_runtime(method):
    import numpy as np
    if method == "grid":
        from vect_grid_scheme import GridScheme
        # Exercise interpolation and the MC operator without using trial data.
        dummy = GridScheme(Ntilde=2, M=16, plot_iter=False)
        dummy.phi(np.zeros((dummy.grid_size, 1)),
                  np.zeros((dummy.grid_size, 1)))
        return None
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.backend.set_floatx("float64")
    tf.keras.utils.set_random_seed(91001)
    dummy = tf.keras.Sequential([
        tf.keras.layers.Input((1,)),
        tf.keras.layers.Dense(21, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(21, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(2),
    ])
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(dummy(tf.ones((32, 1), dtype=tf.float64), training=True)**2)
    for gradient in tape.gradient(loss, dummy.trainable_variables):
        gradient.numpy()
    if method == "picard":
        from NN_Picard_mult_alt import NNPicardSolver
    else:
        from NN_direct_scheme import NNDirecSolver
    tf.keras.backend.clear_session()
    return tf


def run_trial(job):
    process_start = time.perf_counter()
    configure_cpu()
    import numpy as np
    method, budget = job["method"], job["budget"]
    # Imports and runtime work are charged to this trial's separate startup clock.
    with contextlib.redirect_stdout(io.StringIO()):
        tf = initialize_runtime(method)
    startup_seconds = time.perf_counter() - process_start
    np.random.seed(job["seed"])
    random.seed(job["seed"])
    if tf is not None:
        tf.keras.utils.set_random_seed(job["seed"])

    settings = dict(job.get("config", CONFIGS[method][job["config_index"]]))
    solve_start = time.perf_counter()
    deadline = solve_start + budget
    with contextlib.redirect_stdout(io.StringIO()):
        if method == "grid":
            from vect_grid_scheme import GridScheme
            solver = GridScheme(d=1, K_z=0.1, plot_iter=False, **settings)
        elif method == "picard":
            from NN_Picard_mult_alt import NNPicardSolver
            solver = NNPicardSolver(d=1, K_z=0.1, **settings)
        else:
            from NN_direct_scheme import NNDirecSolver
            solver = NNDirecSolver(d=1, K_z=0.1, **settings)
            # The paper uses a 500-update decay interval. An exponential
            # schedule supports an external deadline without constructing a
            # huge piecewise graph or imposing an unrelated step cutoff.
            solver.model.optimizer = tf.keras.optimizers.Adam(
                learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                    solver.initial_learning_rate, 500, solver.lr_decay,
                    staircase=True), epsilon=1e-7)
    setup_seconds = time.perf_counter() - solve_start
    completed_seconds = setup_seconds
    counts = dict(picard_iterations=0, epochs=0, optimizer_steps=0,
                  label_refreshes=0, generated_pairs=0)
    completed_counts = counts.copy()
    if method == "grid":
        u = np.zeros((solver.grid_size, 1))
        ub = np.zeros_like(u)
        while time.perf_counter() < deadline:
            new_u, new_ub = solver.phi(u, ub)
            finished = time.perf_counter()
            if finished > deadline:
                break
            u, ub = new_u, new_ub
            counts["picard_iterations"] += 1
            counts["generated_pairs"] += solver.M * solver.grid_size
            completed_seconds = finished - solve_start
            completed_counts = counts.copy()

        def predict(points):
            return np.concatenate([solver.interpol(points, u),
                                   solver.interpol(points, ub)], axis=1)
    elif method == "picard":
        model = solver.NN
        previous = solver.model_init

        class Deadline(tf.keras.callbacks.Callback):
            """Keep the last complete epoch, including BN moving statistics."""
            def on_epoch_begin(self, epoch, logs=None):
                self.before = model.get_weights()
                if time.perf_counter() >= deadline:
                    model.stop_training = True

            def on_epoch_end(self, epoch, logs=None):
                nonlocal completed_seconds, completed_counts
                finished = time.perf_counter()
                if finished > deadline:
                    model.set_weights(self.before)
                    model.stop_training = True
                else:
                    counts["epochs"] += 1
                    counts["optimizer_steps"] += int(np.ceil(solver.M / solver.batch_size))
                    if epoch == solver.epochs-1:
                        counts["picard_iterations"] += 1
                    completed_seconds = finished - solve_start
                    completed_counts = counts.copy()

        while time.perf_counter() < deadline:
            x = solver.sampleX()
            labels = solver.label(x, previous)
            counts["label_refreshes"] += 1
            counts["generated_pairs"] += solver.M
            if time.perf_counter() >= deadline:
                break
            model.fit(x, labels, epochs=solver.epochs, batch_size=solver.batch_size,
                      shuffle=False, verbose=0, callbacks=[Deadline()])
            if model.stop_training:
                break
            previous = model

        def predict(points):
            return np.concatenate([model(chunk, training=False).numpy()
                                   for chunk in np.array_split(points, max(1, len(points)//4096))])
    else:
        model = solver.model
        step = 0
        inputs, labels = None, None
        while time.perf_counter() < deadline:
            before = solver.NN.get_weights()
            # A checkpoint block never crosses a target-refresh boundary.
            if step % solver.update_frequency == 0:
                inputs = solver.sampleX(num_sample=solver.batch_size)
                labels = solver.label(inputs, model, training=True)
                counts["label_refreshes"] += 1
                counts["generated_pairs"] += solver.batch_size * solver.M
            block = min(10, solver.update_frequency - step % solver.update_frequency)
            for _ in range(block):
                model.train_step(inputs, labels)
            model.optimizer.iterations.numpy()  # Materialize pending updates.
            finished = time.perf_counter()
            if finished > deadline:
                solver.NN.set_weights(before)
                break
            step += block
            counts["optimizer_steps"] = step
            completed_seconds = finished - solve_start
            completed_counts = counts.copy()

        def predict(points):
            return np.concatenate([model(chunk, training=False).numpy()
                                   for chunk in np.array_split(points, max(1, len(points)//4096))])

    execution_seconds = time.perf_counter() - solve_start
    evaluation_start = time.perf_counter()
    result = dict(job)
    result.update(settings=settings, startup_seconds=startup_seconds,
                  setup_seconds=setup_seconds, solve_seconds=completed_seconds,
                  execution_seconds=execution_seconds,
                  overrun_seconds=max(0., execution_seconds-budget),
                  counts=completed_counts,
                  cpu_affinity=sorted(os.sched_getaffinity(0)),
                  numpy_version=np.__version__, tensorflow_version=tf.__version__ if tf else None)
    datasets = [("validation", VALIDATION_SEED, VALIDATION_SIZE)]
    if job["stage"] != "tune":
        datasets.append(("test", TEST_SEED, TEST_SIZE))
    for name, seed, size in datasets:
        points = np.random.default_rng(seed).normal(0., 2., (size, 1))
        prediction = predict(points)
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"Non-finite predictions: {method}, {settings}")
        result[name] = relative_errors(prediction, points)
    result["evaluation_seconds"] = time.perf_counter()-evaluation_start
    result["cold_solve_seconds"] = startup_seconds+completed_seconds
    return result


def run_child(job, destination):
    configure_cpu()
    request = destination.with_suffix(".request.json")
    request.write_text(json.dumps(job))
    child_start = time.perf_counter()
    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                              "--worker", str(request)], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, cwd=ROOT)
    while True:
        try:
            stdout, stderr = child.communicate(timeout=45)
            break
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter()-child_start
            if elapsed > job["budget"]+180:
                child.kill()
                child.communicate()
                raise TimeoutError(f"Worker exceeded its time limit: {job}")
            print(f'  {job["method"]}: worker active for {elapsed:.0f}s '
                  f'(solve budget {job["budget"]:g}s)', flush=True)
    if child.returncode:
        raise RuntimeError(stderr[-6000:]+stdout[-2000:])
    record = json.loads(stdout)
    record["process_seconds"] = time.perf_counter()-child_start
    destination.write_text(json.dumps(record, indent=2))
    request.unlink()
    return record


def manifest(args):
    files = ["compare_schemes.py", "vect_grid_scheme.py", "NN_Picard_mult_alt.py",
             "NN_direct_scheme.py", "experiment_settings.py"]
    return dict(version=VERSION, budgets=args.budgets, tuning_runs=args.tuning_runs,
                runs=args.runs, configs=CONFIGS, validation_size=VALIDATION_SIZE,
                test_size=TEST_SIZE, validation_seed=VALIDATION_SEED,
                test_seed=TEST_SEED, dtype="float64", cpu_threads=1,
                initial_learning_rate=5e-4, picard_lr_decay=.97,
                picard_decay_steps=1000, direct_lr_decay=.6,
                direct_decay_steps=500,
                common_parameters=dict(d=1, c=2., K_z=.1, a=2., b=2.,
                                       theta=1.5, theta_bar=1.5, sig_X=2.),
                nn_architecture=dict(hidden_layers=[21,21], activation="relu",
                                     batch_normalization=True, linear_outputs=2),
                platform=platform.platform(),
                cpu=next((line.split(":",1)[1].strip() for line in
                          Path("/proc/cpuinfo").read_text().splitlines()
                          if line.startswith("model name")), "unknown"),
                code_hashes={f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})


def run_experiment(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specification = manifest(args)
    manifest_path = args.output_dir/"manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != specification:
        raise ValueError("Run settings/source changed; use a different --output-dir.")
    manifest_path.write_text(json.dumps(specification, indent=2))
    records = []
    order = random.Random(1701)

    def perform(job):
        name = f'{job["stage"]}_{job["method"]}_b{job["budget"]:g}_c{job["config_index"]}_s{job["seed"]}.json'
        destination = args.output_dir/name
        if destination.exists():
            row = json.loads(destination.read_text())
        else:
            print(f'Running {name}', flush=True)
            row = run_child(job, destination)
        records.append(row)
        score = row.get("test", row["validation"])["score"]
        print(f'  {row["solve_seconds"]:.2f}s, relative error {score:.4g}', flush=True)

    jobs = [dict(stage="tune", method=method, budget=budget, config_index=index,
                 seed=1100+seed) for method in METHODS for budget in args.budgets
            for index in range(3) for seed in range(args.tuning_runs)]
    order.shuffle(jobs)
    for job in jobs:
        perform(job)
    selected = {method: {} for method in METHODS}
    jobs = []
    for method in METHODS:
        for budget in args.budgets:
            best, scores = choose_config(records, method, budget)
            selected[method][str(budget)] = dict(index=best, validation_means=scores,
                                               settings=CONFIGS[method][best])
            jobs.extend(dict(stage="test", method=method, budget=budget,
                             config_index=best, seed=2100+seed) for seed in range(args.runs))
    (args.output_dir/"selection.json").write_text(json.dumps(selected, indent=2))
    order.shuffle(jobs)
    for job in jobs:
        perform(job)
    (args.output_dir/"results.json").write_text(json.dumps(records, indent=2))
    plot_results(args.output_dir)


def plot_results(directory):
    configure_cpu()
    import numpy as np
    import matplotlib.pyplot as plt
    rows = json.loads((directory/"results.json").read_text())
    budgets = json.loads((directory/"manifest.json").read_text())["budgets"]
    fig, ax = plt.subplots(figsize=(10, 5), dpi=75, layout="tight")
    labels = {"grid": "Grid Picard", "picard": "NN Picard", "direct": "Direct NN"}
    markers = {"grid": "o", "picard": "s", "direct": "^"}
    summary = {}
    for method in METHODS:
        x, y, lo, hi = [], [], [], []
        summary[method] = []
        for budget in budgets:
            data = [r for r in rows if r["stage"] == "test" and r["method"] == method
                    and r["budget"] == budget]
            times = np.array([r["solve_seconds"] for r in data])
            errors = np.array([r["test"]["score"] for r in data])
            if not len(errors):
                raise ValueError("Missing final runs")
            x.append(float(np.median(times)))
            y.append(float(np.median(errors)))
            lo.append(float(np.quantile(errors, .25)))
            hi.append(float(np.quantile(errors, .75)))
            summary[method].append(dict(budget=budget, median_seconds=x[-1],
                                        median_error=y[-1], q25=lo[-1], q75=hi[-1],
                                        runs=len(data)))
        line, = ax.plot(x, y, marker=markers[method], label=labels[method])
        ax.fill_between(x, lo, hi, color=line.get_color(), alpha=.15)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(r"Accuracy vs CPU time, $d=1$, $K_z=0.1$")
    ax.set_xlabel("Elapsed CPU solve time (s)")
    ax.set_ylabel(r"$\max(\Delta u,\,\Delta\bar{u})$")
    ax.legend(frameon=False)
    PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT, dpi=300, bbox_inches="tight")
    fig.savefig(Path(tempfile.gettempdir())/"bsde_scheme_comparison.png", dpi=150)
    plt.close(fig)
    (directory/"summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(directory/"summary.npz", **{m: np.array([[r["median_seconds"],r["median_error"],
                                                    r["q25"],r["q75"]] for r in summary[m]])
                                      for m in METHODS})
    print(f'Saved {PLOT}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Calculate and cache all runs")
    parser.add_argument("--pilot", action="store_true", help="Time one short run per method")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--budgets", nargs="+", type=float, default=[10.,30.,100.,300.])
    parser.add_argument("--tuning-runs", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    if args.worker:
        record = run_trial(json.loads(args.worker.read_text()))
        print(json.dumps(record))
    elif args.pilot:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for method in METHODS:
            job = dict(stage="pilot", method=method, budget=10., config_index=1, seed=7100)
            row = run_child(job, args.output_dir/f"pilot_{method}.json")
            print(json.dumps(row, indent=2), flush=True)
    elif args.run:
        if any(b <= 0 for b in args.budgets) or args.tuning_runs < 1 or args.runs < 1:
            parser.error("Budgets and run counts must be positive")
        run_experiment(args)
    else:
        plot_results(args.output_dir)


if __name__ == "__main__":
    main()
