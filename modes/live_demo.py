"""live_demo.py - makes the arm look alive: it stands straight and gently waves, forever.

    py modes/live_demo.py --port COM7

The two joints follow the same slow sine wave in opposite directions: when the shoulder
leans one way, the elbow bends back the other way. Press a link and it moves away from
your hand, waits, then drifts back. Shows the live torque plot while it runs and the error
plot afterwards. Ctrl+C stops and disables the motors.
"""

import argparse
import math
import multiprocessing
import queue
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # project folder, for dm_motor and the shared files
from dm_motor import Rig
from arm_loop import (ACCELERATION_SMOOTHING, JOINTS, LOOP_HZ, check_safety, plot_errors,
                      wait_until, zero_all_motors)
from error_calculation import calculate_gravity_1, calculate_gravity_2
from live_plot import run_live_plot

# ---- settings -------------------------------------------------------------
SHOULDER_AMPLITUDE_DEG = 5     # shoulder sways between -this and +this
ELBOW_AMPLITUDE_DEG = 10       # elbow sways the opposite way; bigger, so the forearm visibly waves
WAVE_PERIOD_S = 8.0            # seconds for one full wave
FADE_IN_S = 8.0                # the wave grows from nothing over this long, so it starts gently
APPROACH_SPEED_DEG_PER_S = 10  # average speed from the start pose to straight, before waving
KP = 20.0                      # same gains as manual_control.py and the recorded training data
KD = 0.5
PRINT_HZ = 2                   # how many status lines per second

# noise filter: a touch is a smoothed torque error bigger than this many model uncertainties
TOUCH_SCORE_THRESHOLD = 5.0    # lower = more sensitive, but noise starts to trigger it
ERROR_SMOOTHING = 0.1          # 0-1; lower = ignores brief spikes better but reacts later (0.1 = about 50 ms)

# reaction: only the joint of the touched link moves away, on top of the wave
DODGE_DEG = {"shoulder": 15, "elbow": 30}      # how far one touch moves that joint
MAX_DODGE_DEG = {"shoulder": 30, "elbow": 45}  # furthest from the wave if you keep pressing
DODGE_MOVE_S = 0.5             # time to move away
DODGE_HOLD_S = 1.0             # time to wait there
DODGE_RETURN_S = 3.0           # time to drift back
DODGE_SETTLE_S = 0.3           # touches are ignored while moving away and this long after,
                               # because the arm's own quick move looks like a touch
# ---------------------------------------------------------------------------


def ease(x):
    """Goes from 0 to 1 as x goes from 0 to 1, starting and ending with zero speed."""
    return 0.5 - 0.5 * math.cos(math.pi * min(x, 1.0))


def wave_targets(t):
    """Wave angles (rad) t seconds after the wave began. Opposite signs = opposite directions."""
    wave = ease(t / FADE_IN_S) * math.sin(2 * math.pi * t / WAVE_PERIOD_S)
    return {"shoulder": math.radians(SHOULDER_AMPLITUDE_DEG) * wave,
            "elbow": -math.radians(ELBOW_AMPLITUDE_DEG) * wave}


def get_approach_time(start):
    """How long the move from the start pose to straight takes."""
    return max(abs(angle) for angle in start.values()) / math.radians(APPROACH_SPEED_DEG_PER_S)


def get_targets(t, start):
    """Where each joint should be at time t: first move to straight, then wave forever."""
    approach_time = get_approach_time(start)

    if t < approach_time:
        return {name: angle * (1 - ease(t / approach_time)) for name, angle in start.items()}
    return wave_targets(t - approach_time)


class Dodge:
    """One joint's offset from the wave: moves out, holds, then drifts back to 0."""

    def __init__(self):
        self.start_time = -math.inf    # long ago, so the offset starts at 0
        self.move_from = 0.0
        self.move_to = 0.0

    def offset(self, t):
        elapsed = t - self.start_time
        if elapsed < DODGE_MOVE_S:
            return self.move_from + (self.move_to - self.move_from) * ease(elapsed / DODGE_MOVE_S)
        if elapsed < DODGE_MOVE_S + DODGE_HOLD_S:
            return self.move_to
        return self.move_to * (1 - ease((elapsed - DODGE_MOVE_S - DODGE_HOLD_S) / DODGE_RETURN_S))

    def start(self, t, step, limit):
        """Move `step` further from where the joint is now, staying within +-limit."""
        self.move_from = self.offset(t)
        self.move_to = max(-limit, min(limit, self.move_from + step))
        self.start_time = t


def find_press(smoothed_error, uncertainty):
    """Which joint a press turns, or None. A press on link 2 turns the elbow, while a press on
    link 1 can only turn the shoulder, because the elbow is beyond it (same as live_plot.py)."""
    for name in ("elbow", "shoulder"):
        if abs(smoothed_error[name]) > TOUCH_SCORE_THRESHOLD * uncertainty[name]:
            return name
    return None


def run(rig, history, sample_queue, gru):
    not_answering = rig.ensure_feedback()
    if not_answering:
        ids = [motor.can_id for motor in not_answering]
        raise RuntimeError(f"no reply from motor id(s) {ids}")

    zero_all_motors(rig)
    start = rig.positions()

    rig.enable_all()
    rig.bus.tx_gap_s = 0.0    # the SDK's default gap would use up most of a 5 ms tick
    gru.reset_memory()

    period = 1.0 / LOOP_HZ
    ticks_per_print = LOOP_HZ // PRINT_HZ
    start_time = time.perf_counter()
    next_tick = start_time
    tick = 0
    late_ticks = 0

    approach_time = get_approach_time(start)
    dodges = {"shoulder": Dodge(), "elbow": Dodge()}
    smoothed_error = {"shoulder": 0.0, "elbow": 0.0}
    ignore_touches_until = approach_time

    previous_time = None
    previous_velocity = {"shoulder": 0.0, "elbow": 0.0}
    acceleration = {"shoulder": 0.0, "elbow": 0.0}

    while True:
        t = time.perf_counter() - start_time

        rig.spin_once(refresh=False)
        check_safety(rig)

        # acceleration = change in velocity / change in time, lightly smoothed
        if previous_time is not None and t > previous_time:
            for name in acceleration:
                raw = (rig[name].velocity - previous_velocity[name]) / (t - previous_time)
                acceleration[name] += ACCELERATION_SMOOTHING * (raw - acceleration[name])
        previous_time = t
        previous_velocity = {name: rig[name].velocity for name in previous_velocity}

        # compare measured torque with the model's prediction
        gru.feed_live_data(rig["shoulder"].position, rig["elbow"].position,
                           rig["shoulder"].velocity, rig["elbow"].velocity,
                           acceleration["shoulder"], acceleration["elbow"])
        predicted_shoulder = gru.get_predicted_torque_1()
        predicted_elbow = gru.get_predicted_torque_2()

        # touch check: small or brief errors are filtered out here
        error = {"shoulder": rig["shoulder"].torque - predicted_shoulder,
                 "elbow": rig["elbow"].torque - predicted_elbow}
        if gru.is_ready() and t >= ignore_touches_until:
            for name in smoothed_error:
                smoothed_error[name] += ERROR_SMOOTHING * (error[name] - smoothed_error[name])

            pressed = find_press(smoothed_error, {"shoulder": gru.get_uncertainty_1(),
                                                  "elbow": gru.get_uncertainty_2()})
            if pressed:
                # The motor fights the press, so measured torque above predicted means the
                # hand is pushing the joint toward negative. Moving that same way moves away.
                direction = -1 if smoothed_error[pressed] > 0 else 1
                dodges[pressed].start(t, direction * math.radians(DODGE_DEG[pressed]),
                                      math.radians(MAX_DODGE_DEG[pressed]))
                print(f"TOUCH at t={t:.2f}s: {'link 2' if pressed == 'elbow' else 'link 1'} pressed, "
                      f"{pressed} moves toward {'-' if direction < 0 else '+'}")
                ignore_touches_until = t + DODGE_MOVE_S + DODGE_SETTLE_S
                smoothed_error = {"shoulder": 0.0, "elbow": 0.0}

        history["time"].append(t)
        history["shoulder_error"].append(rig["shoulder"].torque - predicted_shoulder)
        history["elbow_error"].append(rig["elbow"].torque - predicted_elbow)

        # send this tick's sample to the live plot; drop it if the plot is behind
        try:
            sample_queue.put_nowait((t,
                                     rig["shoulder"].torque, predicted_shoulder, gru.get_uncertainty_1(),
                                     rig["elbow"].torque, predicted_elbow, gru.get_uncertainty_2()))
        except queue.Full:
            pass

        targets = get_targets(t, start)
        targets_next_tick = get_targets(t + period, start)
        for name in targets:
            targets[name] += dodges[name].offset(t)
            targets_next_tick[name] += dodges[name].offset(t + period)

        gravity = {"shoulder": calculate_gravity_1(targets["shoulder"], targets["elbow"]),
                   "elbow": calculate_gravity_2(targets["shoulder"], targets["elbow"])}

        for name in targets:
            velocity = (targets_next_tick[name] - targets[name]) / period
            rig[name].set_position(targets[name], kp=KP, kd=KD, velocity=velocity, feedforward=gravity[name])

        if tick % ticks_per_print == 0:
            print(f"t={t:7.1f}s   shoulder {math.degrees(rig['shoulder'].position):+6.1f} deg"
                  f"   elbow {math.degrees(rig['elbow'].position):+6.1f} deg   late ticks: {late_ticks}")

        tick += 1
        next_tick += period
        if time.perf_counter() > next_tick:
            late_ticks += 1
            next_tick = time.perf_counter()   # fell behind: carry on instead of catching up
        wait_until(next_tick)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None, help="e.g. COM7")
    args = parser.parse_args()

    # Imported here, not at the top, so the live plot process doesn't load PyTorch.
    from model import gru_prediction as gru
    print(f"using model {gru.load_model()}")

    history = {"time": [], "shoulder_error": [], "elbow_error": []}

    # start the live plot in its own process so drawing never slows the loop
    sample_queue = multiprocessing.Queue(maxsize=2000)
    plot_process = multiprocessing.Process(target=run_live_plot,
                                           args=(sample_queue, TOUCH_SCORE_THRESHOLD), daemon=True)
    plot_process.start()

    with Rig.open(port=args.port, joint_maps=JOINTS) as rig:   # leaving this block disables motors
        try:
            run(rig, history, sample_queue, gru)
        except KeyboardInterrupt:
            print("\nstopped by user")
        except RuntimeError as problem:
            print(f"\nSTOP: {problem}")
        finally:
            rig.disable_all()
            print("motors disabled")
            try:
                sample_queue.put_nowait(None)   # tell the live plot to stop updating
            except queue.Full:
                pass

    # plot only after the motors are off, since plt.show() blocks
    if history["time"]:
        plot_errors(history)


if __name__ == "__main__":
    main()
