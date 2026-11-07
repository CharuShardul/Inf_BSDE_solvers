"""CPU accuracy/time comparison using the three existing Brownian solvers.

Run: python compare_schemes.py --run
Replot versioned reference data: python compare_schemes.py
Add final repeats with the published settings: python compare_schemes.py --extend-runs 5
Short pilot: python compare_schemes.py --pilot
Full protocol (about four CPU hours): python compare_schemes.py --run --full

Only library/runtime warm-up is excluded. Each independent solve includes model
construction, compilation, samples, targets, and optimization. Epochs/blocks are
atomic checkpoints: an update finishing after its budget is rolled back.
"""

import argparse
import ast
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
REFERENCE_DIR = DATA_DIR / "reference"
WORK_DIR = DATA_DIR / "working"
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
VERSION = 2


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
                tuning_seed=args.tuning_seed,
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


def worker_signature(source):
    """Ignore CLI/plot edits when checking whether completed solves can be reused."""
    names = {"configure_cpu", "initialize_runtime", "run_trial", "relative_errors",
             "CONFIGS", "METHODS", "VALIDATION_SEED", "TEST_SEED",
             "VALIDATION_SIZE", "TEST_SIZE"}
    nodes = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(ast.dump(node, include_attributes=False))
        elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in names for target in node.targets):
            nodes.append(ast.dump(node, include_attributes=False))
    return hashlib.sha256("\n".join(nodes).encode()).hexdigest()


def reuse_completed_runs(args, specification):
    """Reuse only identical worker code, solvers, machine, and requested trial keys."""
    source_dir = DATA_DIR/"full_protocol"
    source_file = source_dir/"compare_schemes_source.py"
    if not source_file.exists():
        return
    legacy_path = source_dir/"manifest_original.json"
    if not legacy_path.exists():
        legacy_path = source_dir/"manifest.json"
    legacy = json.loads(legacy_path.read_text())
    old_source = source_file.read_text()
    if hashlib.sha256(source_file.read_bytes()).hexdigest() != legacy["code_hashes"]["compare_schemes.py"]:
        raise ValueError("Archived source does not match its manifest")
    if worker_signature(old_source) != worker_signature(Path(__file__).read_text()):
        return
    if any(legacy["code_hashes"][name] != digest for name,digest in specification["code_hashes"].items()
           if name != "compare_schemes.py"):
        return
    if any(legacy[key] != specification[key] for key in ["cpu", "platform", "dtype", "cpu_threads"]):
        return
    reused = []
    for source in source_dir.glob("tune_*.json"):
        if ".request." in source.name:
            continue
        record = json.loads(source.read_text())
        if record["budget"] not in args.budgets or record["seed"] not in range(
                args.tuning_seed, args.tuning_seed+args.tuning_runs):
            continue
        destination = args.output_dir/source.name
        if not destination.exists():
            record["reused_from"] = str(source.relative_to(ROOT))
            record["source_manifest_sha256"] = hashlib.sha256(legacy_path.read_bytes()).hexdigest()
            destination.write_text(json.dumps(record,indent=2))
        reused.append(source.name)
    if args.output_dir.resolve() == source_dir.resolve():
        current = source_dir/"manifest.json"
        if not (source_dir/"manifest_original.json").exists():
            current.rename(source_dir/"manifest_original.json")
        current.write_text(json.dumps(specification,indent=2))
    (args.output_dir/"cache_reuse.json").write_text(json.dumps(
        dict(worker_signature=worker_signature(old_source), reused=reused,
             original_manifest=str(legacy_path.relative_to(ROOT))),indent=2))
    print(f'Reusing {len(reused)} compatible completed tuning runs.',flush=True)


def run_experiment(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specification = manifest(args)
    # Custom/new working directories start independent experiments. The full
    # legacy protocol alone may resume its previously archived trials.
    if args.output_dir.resolve() == (DATA_DIR/"full_protocol").resolve():
        reuse_completed_runs(args,specification)
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
                 seed=args.tuning_seed+seed) for method in METHODS for budget in args.budgets
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


def plan_additional_runs(records, selected, budgets, total_runs):
    """Add fresh seeds to the fixed, validation-selected configurations."""
    jobs = []
    target_seeds = set(range(2100,2100+total_runs))
    for method in METHODS:
        for budget in budgets:
            group = [r for r in records if r["stage"]=="test"
                     and r["method"]==method and r["budget"]==budget]
            index = selected[method][str(budget)]["index"]
            if any(r["config_index"]!=index for r in group):
                raise ValueError("Existing final runs do not use the frozen selection")
            existing = {r["seed"] for r in group}
            if len(existing)!=len(group) or not existing.issubset(target_seeds):
                raise ValueError("Existing final seeds are duplicated or exceed the target")
            jobs.extend(dict(stage="test",method=method,budget=budget,
                             config_index=index,seed=seed)
                        for seed in sorted(target_seeds-existing))
    return jobs


def extend_reference_runs(args):
    """Run only missing final seeds; do not rerun training-parameter selection."""
    verify_reference(REFERENCE_DIR)
    published = json.loads((REFERENCE_DIR/"manifest.json").read_text())
    if args.budgets!=published["budgets"] or args.runs<published["runs"]:
        raise ValueError("Extend all published budgets and keep at least the existing run count")
    if worker_signature(Path(__file__).read_text())!=published["numerical_worker_sha256"]:
        raise ValueError("The numerical worker changed; run a separate experiment")
    specification = manifest(args)
    if any(specification["code_hashes"][name]!=digest
           for name,digest in published["code_hashes"].items()):
        raise ValueError("Solver sources changed since the published comparison")
    if any(specification[key]!=published[key] for key in ["cpu","platform","dtype","cpu_threads"]):
        raise ValueError("Use the same machine and numerical precision for added repeats")
    selected = json.loads((REFERENCE_DIR/"selection.json").read_text())
    original = json.loads((REFERENCE_DIR/"results.json").read_text())
    jobs = plan_additional_runs(original,selected,args.budgets,args.runs)
    if not jobs:
        print(f'The reference already contains {args.runs} final runs per setting.',flush=True)
        return
    args.output_dir.mkdir(parents=True,exist_ok=True)
    specification.update(extension_of=hashlib.sha256((REFERENCE_DIR/"manifest.json").read_bytes()).hexdigest(),
                         locked_selection_sha256=hashlib.sha256((REFERENCE_DIR/"selection.json").read_bytes()).hexdigest(),
                         numerical_worker_sha256=worker_signature(Path(__file__).read_text()),
                         added_final_runs=len(jobs))
    path = args.output_dir/"manifest.json"
    if path.exists() and json.loads(path.read_text())!=specification:
        raise ValueError("Extension settings/source changed; use a different --output-dir")
    path.write_text(json.dumps(specification,indent=2))
    digest = specification["code_hashes"]["compare_schemes.py"]
    (args.output_dir/f"numerical_source_{digest[:12]}.py").write_bytes(Path(__file__).read_bytes())
    random.Random(1702).shuffle(jobs)
    print(f'Adding {len(jobs)} final runs using the frozen settings; no tuning runs.',flush=True)
    rows = []
    for job in jobs:
        name = f'test_{job["method"]}_b{job["budget"]:g}_c{job["config_index"]}_s{job["seed"]}.json'
        destination = args.output_dir/name
        if destination.exists():
            row = json.loads(destination.read_text())
        else:
            print(f'Running {name}',flush=True)
            row = run_child(job,destination)
        rows.append(row)
        print(f'  {row["solve_seconds"]:.2f}s, relative error {row["test"]["score"]:.4g}',flush=True)
    (args.output_dir/"results.json").write_text(json.dumps(rows,indent=2))
    print('Additional final runs saved; combine them with the reference records to publish.',flush=True)


def verify_reference(directory):
    """Check the frozen data without importing any training dependencies."""
    checksums = json.loads((directory/"checksums.json").read_text())
    for name, expected in checksums.items():
        path = directory/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Reference-data checksum mismatch: {name}")
    return len(checksums)


def plot_results(directory, *, summary_dir=None):
    configure_cpu()
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch
    if directory.resolve() == REFERENCE_DIR.resolve() and (directory/"checksums.json").exists():
        verify_reference(directory)
    # Keep the versioned reference immutable during routine plotting.
    if summary_dir is None:
        summary_dir = DATA_DIR/"rendered" if directory.resolve() == REFERENCE_DIR.resolve() else directory
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows = json.loads((directory/"results.json").read_text())
    budgets = json.loads((directory/"manifest.json").read_text())["budgets"]
    fig, ax = plt.subplots(figsize=(10, 5), dpi=75, layout="tight")
    labels = {"grid": "Grid Picard", "picard": "NN Picard", "direct": "Direct NN"}
    markers = {"grid": "o", "picard": "s", "direct": "^"}
    summary = {}
    legend_handles = []
    run_counts = set()
    for method in METHODS:
        x, y, lo, hi = [], [], [], []
        run_times, run_errors = [], []
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
            run_times.append(times)
            run_errors.append(errors)
            run_counts.add(len(data))
            summary[method].append(dict(budget=budget, median_seconds=x[-1],
                                        median_error=y[-1], q25=lo[-1], q75=hi[-1],
                                        runs=len(data)))
        line, = ax.plot(x, y, linewidth=1., alpha=.6, zorder=1)
        color = line.get_color()
        # Widths scale with elapsed time to stay comparable on the log axis.
        # Each box uses every final-run error, not precomputed quartiles.
        ax.boxplot(run_errors, positions=x, widths=.2*np.asarray(x),
                   manage_ticks=False, patch_artist=True, whis=1.5,
                   showfliers=False,
                   boxprops=dict(facecolor=to_rgba(color,.22), edgecolor=color),
                   medianprops=dict(color="black", linewidth=1.3),
                   whiskerprops=dict(color=color), capprops=dict(color=color))
        for times, errors in zip(run_times, run_errors):
            ax.scatter(times, errors, marker=markers[method], s=20,
                       color=color, edgecolors="white", linewidths=.3, zorder=4)
        legend_handles.append(Patch(facecolor=to_rgba(color,.22),
                                    edgecolor=color, label=labels[method]))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(budgets, labels=[f"{budget:g}" for budget in budgets])
    ax.set_title(r"Accuracy vs CPU time, $d=1$, $K_z=0.1$")
    ax.set_xlabel("Elapsed CPU solve time (s)")
    ax.set_ylabel(r"$\max(\Delta u,\,\Delta\bar{u})$")
    legend_title = (f"{next(iter(run_counts))} runs per box" if len(run_counts)==1
                    else "Boxes: IQR; points: individual runs")
    ax.legend(handles=legend_handles, title=legend_title, frameon=False)
    PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT, dpi=300, bbox_inches="tight",
                metadata={"CreationDate":None, "ModDate":None})
    fig.savefig(Path(tempfile.gettempdir())/"bsde_scheme_comparison.png", dpi=150)
    plt.close(fig)
    (summary_dir/"summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(summary_dir/"summary.npz", **{m: np.array([[r["median_seconds"],r["median_error"],
                                                    r["q25"],r["q75"]] for r in summary[m]])
                                      for m in METHODS})
    (summary_dir/"plot_metadata.json").write_text(json.dumps(dict(
        style="boxplots", box_percentiles=[25,75], whisker_iqr_multiple=1.5,
        individual_run_points=True, runs_per_box=sorted(run_counts),
        box_position="median elapsed solve time",
        numerical_manifest_sha256=hashlib.sha256((directory/"manifest.json").read_bytes()).hexdigest(),
        results_sha256=hashlib.sha256((directory/"results.json").read_bytes()).hexdigest(),
        rendering_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ),indent=2))
    print(f'Saved {PLOT}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Calculate and cache all runs")
    parser.add_argument("--extend-runs",type=int,help="Add final repeats using the reference settings")
    parser.add_argument("--full", action="store_true", help="Use the original four-hour, five-run protocol")
    parser.add_argument("--pilot", action="store_true", help="Time one short run per method")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--budgets", nargs="+", type=float)
    parser.add_argument("--tuning-runs", type=int)
    parser.add_argument("--tuning-seed", type=int)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    args.budgets = args.budgets if args.budgets is not None else ([10.,30.,100.,300.] if args.full else [5.,10.,20.,30.,100.])
    args.tuning_runs = args.tuning_runs if args.tuning_runs is not None else (2 if args.full else 1)
    args.tuning_seed = args.tuning_seed if args.tuning_seed is not None else (1100 if args.full else 1101)
    args.runs = args.extend_runs if args.extend_runs is not None else (args.runs if args.runs is not None else 5)
    if args.extend_runs is not None and (args.run or args.pilot or args.full):
        parser.error("Use --extend-runs separately from --run, --pilot, and --full")
    if args.output_dir is None:
        if args.extend_runs is not None:
            args.output_dir = DATA_DIR/f"extension_{args.runs}_runs"
        elif args.full:
            args.output_dir = DATA_DIR/"full_protocol"
        elif args.run or args.pilot:
            args.output_dir = WORK_DIR
        else:
            args.output_dir = REFERENCE_DIR if (REFERENCE_DIR/"results.json").exists() else DATA_DIR
    args.output_dir = args.output_dir.resolve()
    if (args.run or args.pilot or args.extend_runs is not None) and args.output_dir == REFERENCE_DIR.resolve():
        parser.error("The reference data is frozen; use a separate --output-dir for new runs")
    if args.worker:
        record = run_trial(json.loads(args.worker.read_text()))
        print(json.dumps(record))
    elif args.extend_runs is not None:
        extend_reference_runs(args)
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
