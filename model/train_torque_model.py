"""train_torque_model.py - teach a small GRU to predict motor torque from arm motion.

    py model/train_torque_model.py

Reads every run in data/ (made by collect_torque_data.py).
    Inputs,  each tick:  position, velocity, acceleration for shoulder and elbow (6 numbers)
    Outputs, each tick:  predicted torque and its uncertainty for shoulder and elbow (4 numbers)
Model: GRU with 64 units -> GRU with 32 units -> linear layer to the 4 outputs.

Training first uses plain squared error for a few epochs, so the torque prediction is
already decent, then GaussianNLLLoss, which also teaches the uncertainty: how far off the
prediction is likely to be at this moment. The loss is lowest when the uncertainty matches
the real size of the errors, and it can go below zero, which is normal.

The last 20% of each run is held back for validation. It is one unbroken block, not random
rows, because neighbouring rows are 5 ms apart and nearly identical: random rows would let
the model be tested on data it has effectively already seen.

Saves model/trained/torque_gru_<date>.pt (the model and its scaling) and a .png of predicted
against measured torque on the validation data. Don't train while recording, because
training uses the whole CPU and would upset the 200 Hz loop.
"""

import argparse
import copy
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

JOINTS = ["shoulder", "elbow"]
INPUT_COLUMNS = [f"{joint}_{value}" for joint in JOINTS for value in ("q_rad", "dq_rad_s", "ddq_rad_s2")]
OUTPUT_COLUMNS = [f"{joint}_tau_nm" for joint in JOINTS]

# ---- settings -------------------------------------------------------------
LOOP_HZ = 200
SEQUENCE_TICKS = 200       # length of each training piece (1 s)
WARM_UP_TICKS = 50         # the model gets this long to settle before its answers are scored
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
SQUARED_ERROR_EPOCHS = 10  # epochs of plain squared error before the uncertainty is trained
MAX_EPOCHS = 100           # most epochs of GaussianNLLLoss after that
MIN_UNCERTAINTY_NM = 1e-3  # keeps the uncertainty above zero
STOP_AFTER_NO_BETTER = 12  # stop when validation hasn't improved for this many epochs
VALIDATION_FRACTION = 0.2
MINIMUM_RUN_TICKS = 5 * SEQUENCE_TICKS
PLOT_SECONDS = 30
TOUCH_SCORE_THRESHOLD = 4.0  # a touch is |measured - predicted| / uncertainty above this
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
# ---------------------------------------------------------------------------


class TorqueGRU(nn.Module):
    """Takes real units and returns N*m. The scaling is stored with the model, so a saved
    model can be used later without remembering how the training data was scaled.

    Models trained before the uncertainty was added use has_uncertainty=False and give
    nan for it."""

    def __init__(self, input_mean, input_std, output_mean, output_std, has_uncertainty=True):
        super().__init__()
        self.has_uncertainty = has_uncertainty
        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_std", input_std)
        self.register_buffer("output_mean", output_mean)
        self.register_buffer("output_std", output_std)
        self.gru_64 = nn.GRU(len(INPUT_COLUMNS), 64, batch_first=True)
        self.gru_32 = nn.GRU(64, 32, batch_first=True)
        self.to_torque = nn.Linear(32, len(OUTPUT_COLUMNS) * (2 if has_uncertainty else 1))

    def forward(self, inputs, memory=None):
        """inputs: (batches, ticks, 6). Pass memory back in to carry on from the last call.
        Returns torque, uncertainty (one standard deviation) and memory."""
        memory_64, memory_32 = memory if memory is not None else (None, None)
        scaled = (inputs - self.input_mean) / self.input_std
        hidden, memory_64 = self.gru_64(scaled, memory_64)
        hidden, memory_32 = self.gru_32(hidden, memory_32)
        outputs = self.to_torque(hidden)
        joints = len(OUTPUT_COLUMNS)
        torque = outputs[..., :joints] * self.output_std + self.output_mean
        if self.has_uncertainty:
            uncertainty = nn.functional.softplus(outputs[..., joints:]) * self.output_std + MIN_UNCERTAINTY_NM
        else:
            uncertainty = torch.full_like(torque, float("nan"))
        return torque, uncertainty, (memory_64, memory_32)


def load_run(path):
    """The motion part of one recording, as (inputs, outputs) arrays."""
    with open(path, newline="") as file:
        reader = csv.DictReader(file)
        missing = [c for c in INPUT_COLUMNS + OUTPUT_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{path.name} has no {missing[0]} column. It was recorded before "
                             "acceleration was added: re-record it or move it out of the folder.")
        rows = [row for row in reader if row["phase"] == "motion"]
    inputs = np.array([[float(row[c]) for c in INPUT_COLUMNS] for row in rows], dtype=np.float32)
    outputs = np.array([[float(row[c]) for c in OUTPUT_COLUMNS] for row in rows], dtype=np.float32)
    return inputs.reshape(-1, len(INPUT_COLUMNS)), outputs.reshape(-1, len(OUTPUT_COLUMNS))


def load_all_runs(folder):
    """Split each run in time into training and validation pieces."""
    train, validation = [], []
    for path in sorted(folder.glob("run_*.csv")):
        inputs, outputs = load_run(path)
        print(f"  {path.name}: {len(inputs) / LOOP_HZ / 60:.1f} min of motion")
        if len(inputs) < MINIMUM_RUN_TICKS:
            print("    too short, skipped")
            continue
        split = int(len(inputs) * (1 - VALIDATION_FRACTION))
        train.append((inputs[:split], outputs[:split]))
        validation.append((inputs[split:], outputs[split:]))
    if not train:
        raise SystemExit(f"no usable recordings in {folder}")
    return train, validation


def random_batch(runs, rng):
    """BATCH_SIZE pieces of SEQUENCE_TICKS, each from a random place in a random run."""
    lengths = np.array([len(inputs) for inputs, _ in runs], dtype=float)
    picks = rng.choice(len(runs), size=BATCH_SIZE, p=lengths / lengths.sum())
    batch_inputs, batch_outputs = [], []
    for pick in picks:
        inputs, outputs = runs[pick]
        start = rng.integers(0, len(inputs) - SEQUENCE_TICKS + 1)
        batch_inputs.append(inputs[start:start + SEQUENCE_TICKS])
        batch_outputs.append(outputs[start:start + SEQUENCE_TICKS])
    return torch.from_numpy(np.stack(batch_inputs)), torch.from_numpy(np.stack(batch_outputs))


def predict_runs(model, runs):
    """Run the model straight through each validation block, as it would run live.
    Returns one (predicted, uncertainty, measured) set per run, without the warm-up ticks."""
    results = []
    model.eval()
    with torch.no_grad():
        for inputs, outputs in runs:
            predicted, uncertainty, _ = model(torch.from_numpy(inputs).unsqueeze(0))
            results.append((predicted[0, WARM_UP_TICKS:].numpy(),
                            uncertainty[0, WARM_UP_TICKS:].numpy(),
                            outputs[WARM_UP_TICKS:]))
    model.train()
    return results


def squared_error_loss(model, predicted, uncertainty, measured):
    return (((predicted - measured) / model.output_std) ** 2).mean()


def gaussian_loss(model, predicted, uncertainty, measured):
    """Torque scaled by output_std, so both joints count equally despite their sizes."""
    return nn.functional.gaussian_nll_loss(predicted / model.output_std, measured / model.output_std,
                                           (uncertainty / model.output_std) ** 2)


def validation_loss(model, runs, loss_function):
    results = predict_runs(model, runs)
    predicted, uncertainty, measured = (torch.from_numpy(np.concatenate([r[i] for r in results]))
                                        for i in range(3))
    with torch.no_grad():
        return loss_function(model, predicted, uncertainty, measured).item()


def train(model, train_runs, validation_runs, epochs, rng):
    print("squared error first, to get the torque prediction close")
    train_stage(model, train_runs, validation_runs, SQUARED_ERROR_EPOCHS, rng, squared_error_loss)
    print("GaussianNLLLoss, to train the uncertainty as well (lower is better, can go negative)")
    train_stage(model, train_runs, validation_runs, epochs, rng, gaussian_loss)


def train_stage(model, train_runs, validation_runs, epochs, rng, loss_function):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)

    # one epoch scores about as many ticks as there are in the training data
    train_ticks = sum(len(inputs) for inputs, _ in train_runs)
    batches_per_epoch = max(1, train_ticks // (BATCH_SIZE * (SEQUENCE_TICKS - WARM_UP_TICKS)))

    best_loss = float("inf")
    best_state = None
    epochs_without_better = 0
    for epoch in range(1, epochs + 1):
        total = 0.0
        for _ in range(batches_per_epoch):
            inputs, outputs = random_batch(train_runs, rng)
            predicted, uncertainty, _ = model(inputs)
            loss = loss_function(model, predicted[:, WARM_UP_TICKS:], uncertainty[:, WARM_UP_TICKS:],
                                 outputs[:, WARM_UP_TICKS:])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()

        val = validation_loss(model, validation_runs, loss_function)
        scheduler.step(val)
        print(f"epoch {epoch:3d}  train {total / batches_per_epoch:.4f}  validation {val:.4f}")

        if val < best_loss:
            best_loss = val
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_better = 0
        else:
            epochs_without_better += 1
            if epochs_without_better >= STOP_AFTER_NO_BETTER:
                print("validation stopped improving")
                break
    model.load_state_dict(best_state)


def report(results):
    """Prediction error per joint. Near-zero R2 means no better than guessing the average.

    Also checks the uncertainty. If it's honest, about 68%, 95% and 99.7% of errors fall
    within 1, 2 and 3 uncertainties. Nobody touched the arm in this data, so ticks over
    the touch score are false alarms."""
    predicted = np.concatenate([r[0] for r in results])
    uncertainty = np.concatenate([r[1] for r in results])
    measured = np.concatenate([r[2] for r in results])
    numbers = {}
    print("\nvalidation data (never trained on):")
    for index, joint in enumerate(JOINTS):
        error = predicted[:, index] - measured[:, index]
        rmse = float(np.sqrt((error ** 2).mean()))
        r2 = float(1 - (error ** 2).sum() / ((measured[:, index] - measured[:, index].mean()) ** 2).sum())
        p99 = float(np.percentile(np.abs(error), 99))
        score = np.abs(error) / uncertainty[:, index]
        within = [float((score < n).mean()) for n in (1, 2, 3)]
        false_alarms = float((score > TOUCH_SCORE_THRESHOLD).mean())
        numbers[joint] = {"rmse_nm": rmse, "r2": r2, "abs_error_p99_nm": p99,
                          "uncertainty_median_nm": float(np.median(uncertainty[:, index])),
                          "uncertainty_min_nm": float(uncertainty[:, index].min()),
                          "within_1_2_3_uncertainties": within,
                          "ticks_over_touch_score": false_alarms,
                          "score_max": float(score.max())}
        print(f"  {joint:8s}  typical error (RMSE) {rmse:.3f} N*m   R2 {r2:.3f}   99% of errors under {p99:.3f} N*m")
        print(f"            uncertainty median {np.median(uncertainty[:, index]):.3f} N*m, "
              f"within 1/2/3 uncertainties: {within[0]:.1%} / {within[1]:.1%} / {within[2]:.1%} "
              f"(ideal 68.3 / 95.4 / 99.7)")
        print(f"            ticks over touch score {TOUCH_SCORE_THRESHOLD:g}: {false_alarms:.3%}, "
              f"highest score {score.max():.1f}")
    return numbers


def plot(results, path):
    predicted, uncertainty, measured = results[0]
    ticks = min(len(predicted), PLOT_SECONDS * LOOP_HZ)
    time_s = np.arange(ticks) / LOOP_HZ
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 6))
    for index, (axis, joint) in enumerate(zip(axes, JOINTS)):
        axis.plot(time_s, measured[:ticks, index], label="measured", linewidth=1)
        axis.plot(time_s, predicted[:ticks, index], label="predicted", linewidth=1)
        axis.fill_between(time_s,
                          predicted[:ticks, index] - 3 * uncertainty[:ticks, index],
                          predicted[:ticks, index] + 3 * uncertainty[:ticks, index],
                          color="tab:orange", alpha=0.2, linewidth=0, label="predicted +- 3 uncertainties")
        axis.set_ylabel(f"{joint} torque (N*m)")
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("time in validation data (s)")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Train the torque-prediction GRU.")
    parser.add_argument("--data", type=Path, default=PROJECT / "data")
    parser.add_argument("--output", type=Path, default=HERE / "trained")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"loading {args.data}")
    train_runs, validation_runs = load_all_runs(args.data)

    # scaling comes from the training data only, so validation stays unseen
    train_inputs = np.concatenate([inputs for inputs, _ in train_runs])
    train_outputs = np.concatenate([outputs for _, outputs in train_runs])
    print(f"training on {len(train_inputs) / LOOP_HZ / 60:.1f} min, "
          f"validating on {sum(len(i) for i, _ in validation_runs) / LOOP_HZ / 60:.1f} min")

    model = TorqueGRU(torch.from_numpy(train_inputs.mean(axis=0)),
                      torch.from_numpy(train_inputs.std(axis=0) + 1e-6),
                      torch.from_numpy(train_outputs.mean(axis=0)),
                      torch.from_numpy(train_outputs.std(axis=0) + 1e-6))
    print(f"model has {sum(p.numel() for p in model.parameters())} parameters")

    train(model, train_runs, validation_runs, args.epochs, rng)

    results = predict_runs(model, validation_runs)
    numbers = report(results)

    args.output.mkdir(exist_ok=True)
    stem = datetime.now().strftime("torque_gru_%Y%m%d_%H%M%S")
    model_path = args.output / f"{stem}.pt"
    plot_path = args.output / f"{stem}.png"
    torch.save({"state_dict": model.state_dict(),
                "input_columns": INPUT_COLUMNS,
                "output_columns": OUTPUT_COLUMNS,
                "loop_hz": LOOP_HZ,
                "validation": numbers}, model_path)
    plot(results, plot_path)
    print(f"\nsaved {model_path}\nsaved {plot_path}")


if __name__ == "__main__":
    main()
