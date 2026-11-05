"""gru_prediction.py - torque prediction from the trained GRU, for use inside a live loop.

Works like error_calculation.py, with one difference: the GRU remembers the past, so
feeding it data and reading its answer are separate steps. Feed it exactly once per tick
at 200 Hz, the rate it was trained at. Read the answer as many times as you like.

    from model.gru_prediction import (load_model, reset_memory, feed_live_data,
                                get_predicted_torque_1, get_predicted_torque_2)

    load_model()       # newest model in model/trained/; do this before the timed loop starts
    reset_memory()     # at the start of each run

    # every tick:
    feed_live_data(shoulder_position, elbow_position,
                   shoulder_velocity, elbow_velocity,
                   shoulder_acceleration, elbow_acceleration)
    shoulder_error = rig["shoulder"].torque - get_predicted_torque_1()
    elbow_error = rig["elbow"].torque - get_predicted_torque_2()

    # how unusual that error is, and whether it counts as a touch:
    shoulder_score = get_touch_score_1(rig["shoulder"].torque)
    elbow_score = get_touch_score_2(rig["elbow"].torque)
    touched = shoulder_score > TOUCH_SCORE_THRESHOLD or elbow_score > TOUCH_SCORE_THRESHOLD

The uncertainty is the model's own estimate of how far off it is likely to be right now, as
one standard deviation in N*m. The score is the error measured in those units, so 4 means
four times further off than the model expected.

Inputs are measured values in rad, rad/s and rad/s^2. Work out acceleration the way
collect_torque_data.py does, with the same ACCELERATION_SMOOTHING, or the model will see
something different from its training data. Never pass commanded values.

The torque, uncertainty and score functions return nan until the GRU has had WARM_UP_TICKS of data (0.25 s),
because its early answers were not part of training and can't be trusted. Comparisons
with nan are always False, so a threshold check won't report a touch during that time.
Models trained before the uncertainty was added still predict torque, but their uncertainty
and score are always nan, so they never report a touch.
"""

import math
from pathlib import Path

import torch
import time

from model.train_torque_model import INPUT_COLUMNS, TOUCH_SCORE_THRESHOLD, WARM_UP_TICKS, TorqueGRU

MODEL_FOLDER = Path(__file__).resolve().parent / "trained"

_model = None
_memory = None
_ticks_fed = 0
_predicted_torque = (math.nan, math.nan)
_uncertainty = (math.nan, math.nan)


def load_model(path=None):
    """Load a model saved by train_torque_model.py. With no path, uses the newest in model/trained/."""
    global _model
    if path is None:
        saved = sorted(MODEL_FOLDER.glob("torque_gru_*.pt"))
        if not saved:
            raise FileNotFoundError(f"no trained model in {MODEL_FOLDER}; run model/train_torque_model.py first")
        path = saved[-1]

    checkpoint = torch.load(path)
    if checkpoint["input_columns"] != INPUT_COLUMNS:
        raise ValueError(f"{path} expects inputs {checkpoint['input_columns']}, not {INPUT_COLUMNS}")

    state = checkpoint["state_dict"]
    has_uncertainty = state["to_torque.weight"].shape[0] == 2 * len(checkpoint["output_columns"])
    if not has_uncertainty:
        print(f"{path} has no uncertainty output, so touches won't be reported. Retrain to get one.")
    model = TorqueGRU(state["input_mean"], state["input_std"], state["output_mean"], state["output_std"],
                      has_uncertainty=has_uncertainty)
    model.load_state_dict(state)
    model.eval()

    torch.set_num_threads(1)   # extra threads only add delay for one tick of a small model
    _model = model
    reset_memory()
    return path


def reset_memory():
    """Forget the past. Call at the start of a run, or after any gap in the data."""
    global _memory, _ticks_fed, _predicted_torque, _uncertainty
    _memory = None
    _ticks_fed = 0
    _predicted_torque = (math.nan, math.nan)
    _uncertainty = (math.nan, math.nan)


def feed_live_data(theta1, theta2,
                   theta1_velocity, theta2_velocity,
                   theta1_acceleration, theta2_acceleration):
    """Give the GRU this tick's measurements and work out its new prediction."""
    global _memory, _ticks_fed, _predicted_torque, _uncertainty
    if _model is None:
        load_model()

    inputs = torch.tensor([[[theta1, theta1_velocity, theta1_acceleration,
                             theta2, theta2_velocity, theta2_acceleration]]], dtype=torch.float32)
    with torch.inference_mode():
        time_before = time.time()    
        torque, uncertainty, _memory = _model(inputs, _memory)
        #print("TIME FOR RUN: ", time.time() - time_before)

    _ticks_fed += 1
    if is_ready():
        _predicted_torque = (torque[0, 0, 0].item(), torque[0, 0, 1].item())
        _uncertainty = (uncertainty[0, 0, 0].item(), uncertainty[0, 0, 1].item())


def is_ready():
    """True once the GRU has had enough data for its predictions to be used."""
    return _ticks_fed >= WARM_UP_TICKS


def get_predicted_torque_1():
    """Predicted shoulder torque (N*m) for the last tick fed in, or nan while warming up."""
    return _predicted_torque[0]


def get_predicted_torque_2():
    """Predicted elbow torque (N*m) for the last tick fed in, or nan while warming up."""
    return _predicted_torque[1]


def get_uncertainty_1():
    """How far off the shoulder prediction is likely to be (N*m, one standard deviation)."""
    return _uncertainty[0]


def get_uncertainty_2():
    """How far off the elbow prediction is likely to be (N*m, one standard deviation)."""
    return _uncertainty[1]


def get_touch_score_1(measured_torque):
    """Shoulder error measured in uncertainties. Compare with TOUCH_SCORE_THRESHOLD."""
    return abs(measured_torque - _predicted_torque[0]) / _uncertainty[0]


def get_touch_score_2(measured_torque):
    """Elbow error measured in uncertainties. Compare with TOUCH_SCORE_THRESHOLD."""
    return abs(measured_torque - _predicted_torque[1]) / _uncertainty[1]
