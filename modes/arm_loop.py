"""arm_loop.py - two-joint arm with a timed main loop.

    py modes/arm_loop.py --port COM7

Every tick:  read feedback  ->  decide targets  ->  send commands  ->  wait for next tick
Ctrl+C stops and disables the motors.
"""

import argparse
import math
import multiprocessing
import queue
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # project folder, for dm_motor, model and the shared files
from dm_motor import JointMap, MotorType, Rig
from live_plot import run_live_plot
# ---- settings -------------------------------------------------------------
# The motors face opposite ways. `sign` is the ONLY place that is handled: the SDK
# flips position, velocity, measured torque and every outgoing command, so all
# other code works in the model's convention:
#     shoulder 0 = link 1 straight up, +-90 deg = horizontal, + tilts toward one chosen side
#     elbow 0 = fully straight, + turns the same way as + shoulder
# Run check_directions.py to confirm these two values on the hardware.
JOINTS = [
    JointMap("shoulder", can_id=1, motor_type=MotorType.DM_J4310, master_id=0, sign=-1),
    JointMap("elbow",    can_id=2, motor_type=MotorType.DM_J4310, master_id=0, sign=+1),
]
JOINT_LIMIT_DEG = 60       # targets stay within -this..+this; shoulder 0 = straight up, elbow 0 = straight
LOOP_HZ = 200              # how many times per second the loop runs
PRINT_HZ = 5               # how many status lines per second
KP = 5.0                   # stiffness
KD = 0.5                   # damping
MAX_FEEDBACK_AGE = 0.1     # seconds; older than this means a motor stopped answering
ACCELERATION_SMOOTHING = 0.1   # 0-1; lower = smoother but slower acceleration estimate
ZERO_AT_STARTUP = True         # zero both motors at a known pose before running
SWING_AMPLITUDE_DEG = 20       # joints swing between -this and +this
SWING_SPEED_DEG_PER_S = 20     # constant joint speed while swinging
# ---------------------------------------------------------------------------


def move_toward(current, goal, max_step):
    """Step from current toward goal by at most max_step."""
    if goal > current:
        return min(current + max_step, goal)
    return max(current - max_step, goal)


def triangle_wave(time_in_wave):
    """Angle (rad) moving at constant speed: 0 -> +20 -> -20 -> 0, repeating."""
    amplitude = math.radians(SWING_AMPLITUDE_DEG)
    speed = math.radians(SWING_SPEED_DEG_PER_S)
    period = 4 * amplitude / speed                    # 4 s for 20 deg at 20 deg/s

    phase = (time_in_wave % period) / period          # 0 to 1 through one cycle
    if phase < 0.25:                                  # 0 up to +20
        return amplitude * (phase / 0.25)
    if phase < 0.75:                                  # +20 down to -20
        return amplitude * (1 - (phase - 0.25) / 0.25)
    return amplitude * ((phase - 0.75) / 0.25 - 1)    # -20 back up to 0


def get_targets(t, start):
    """Where each joint should be at time t (seconds since start).

    1. Approach: both joints move from their start angle to 0 deg at
       SWING_SPEED_DEG_PER_S, so nothing jumps at startup.
    2. Swing: both joints oscillate between -20 and +20 deg together,
       at a constant SWING_SPEED_DEG_PER_S.
    """
    speed = math.radians(SWING_SPEED_DEG_PER_S)

    # time for the joint that starts furthest from 0 to get there
    approach_time = max(abs(angle) for angle in start.values()) / speed

    if t < approach_time:
        return {name: move_toward(angle, 0.0, speed * t) for name, angle in start.items()}

    swing_angle = triangle_wave(t - approach_time)
    return {"shoulder": swing_angle, "elbow": swing_angle}


def check_safety(rig):
    """Raise an error if any motor reports a fault or has stopped answering."""
    for joint in rig.joints:
        status = joint.telemetry()
        if not status.ok:
            raise RuntimeError(f"{joint.name}: motor fault ({status.status_text})")
        if status.age_s > MAX_FEEDBACK_AGE:
            raise RuntimeError(f"{joint.name}: no feedback for {status.age_s * 1000:.0f} ms")


def print_status(t, rig, late_ticks, scores):
    line = f"t={t:6.2f}s"
    for joint in rig.joints:
        line += (f"   {joint.name}: {math.degrees(joint.position):+7.2f} deg  {joint.torque:+.3f} Nm"
                 f"  score {scores[joint.name]:4.1f}")
    line += f"   late ticks: {late_ticks}"
    print(line)


def wait_until(deadline):
    """Sleep until `deadline` (a time.perf_counter() value).

    time.sleep on Windows can oversleep by several ms, so sleep most of the
    way and then spin for the last bit.
    """
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        if remaining > 0.002:
            time.sleep(remaining - 0.0015)


def zero_all_motors(rig):
    """Set every motor's current position as its zero (motors must be disabled)."""
    input("\nZEROING: hold the arm with link 1 STRAIGHT UP and the elbow STRAIGHT,\n"
          "keep it still, then press Enter...")

    for joint in rig.joints:
        joint.motor.set_zero()
        time.sleep(0.02)

    # ask each motor for a brand-new reading, then check the new zero took
    time.sleep(0.05)
    for joint in rig.joints:
        joint.refresh()
        time.sleep(0.02)
    time.sleep(0.05)
    rig.spin_once(refresh=False)

    for joint in rig.joints:
        angle_deg = math.degrees(joint.position)
        if abs(angle_deg) > 1.0:
            raise RuntimeError(f"{joint.name}: zero did not take (reads {angle_deg:+.1f} deg)")
    print("zeroed both motors")


def run(rig, history, sample_queue, gru):
    # 1. Make sure every motor is answering, then remember where they are.
    not_answering = rig.ensure_feedback()
    if not_answering:
        ids = [motor.can_id for motor in not_answering]
        raise RuntimeError(f"no reply from motor id(s) {ids}")

    # Zero both motors at a known pose, so 0 deg means what the dynamics model expects.
    if ZERO_AT_STARTUP:
        zero_all_motors(rig)

    start = rig.positions()
    print("start pose (deg):", {name: round(math.degrees(q), 1) for name, q in start.items()})

    # 2. Turn the motors on.
    rig.enable_all()

    # The SDK sleeps 3 ms after every frame by default, which would use up most of a 5 ms
    # tick. The GRU was trained at 200 Hz, so the loop needs to actually run at 200 Hz.
    rig.bus.tx_gap_s = 0.0
    gru.reset_memory()

    # 3. Timed loop.
    period = 1.0 / LOOP_HZ
    ticks_per_print = LOOP_HZ // PRINT_HZ
    start_time = time.perf_counter()
    next_tick = start_time
    tick = 0
    late_ticks = 0
    was_touched = False

    previous_time = None
    previous_shoulder_velocity = 0
    previous_elbow_velocity = 0
    shoulder_acceleration = 0
    elbow_acceleration = 0

    while True:
        t = time.perf_counter() - start_time

        # read: collect the replies the motors sent after the last command
        rig.spin_once(refresh=False)
        check_safety(rig)

        shoulder_position = rig["shoulder"].position
        elbow_position = rig["elbow"].position
        shoulder_velocity = rig["shoulder"].velocity
        elbow_velocity = rig["elbow"].velocity

        # acceleration = change in velocity / change in time, lightly smoothed
        if previous_time is not None:
            dt = t - previous_time
            if dt > 0:
                raw_shoulder_acceleration = (shoulder_velocity - previous_shoulder_velocity) / dt
                raw_elbow_acceleration = (elbow_velocity - previous_elbow_velocity) / dt
                shoulder_acceleration += ACCELERATION_SMOOTHING * (raw_shoulder_acceleration - shoulder_acceleration)
                elbow_acceleration += ACCELERATION_SMOOTHING * (raw_elbow_acceleration - elbow_acceleration)

        previous_time = t
        previous_shoulder_velocity = shoulder_velocity
        previous_elbow_velocity = elbow_velocity

        # decide
        limit = math.radians(JOINT_LIMIT_DEG)
        targets = {name: max(-limit, min(limit, angle)) for name, angle in get_targets(t, start).items()}

        # find errors
        gru.feed_live_data(shoulder_position, elbow_position,
                           shoulder_velocity, elbow_velocity,
                           shoulder_acceleration, elbow_acceleration)
        predicted_shoulder_torque = gru.get_predicted_torque_1()
        predicted_elbow_torque = gru.get_predicted_torque_2()

        shoulder_error = rig["shoulder"].torque - predicted_shoulder_torque
        elbow_error = rig["elbow"].torque - predicted_elbow_torque

        # touch check: how many uncertainties away from the prediction each joint is
        scores = {"shoulder": gru.get_touch_score_1(rig["shoulder"].torque),
                  "elbow": gru.get_touch_score_2(rig["elbow"].torque)}
        touched = any(score > gru.TOUCH_SCORE_THRESHOLD for score in scores.values())
        if touched and not was_touched:
            print(f"TOUCH at t={t:.2f}s  score: shoulder {scores['shoulder']:.1f}, elbow {scores['elbow']:.1f}")
        was_touched = touched

        history["time"].append(t)
        history["shoulder_error"].append(shoulder_error)
        history["elbow_error"].append(elbow_error)

        # send this tick's sample to the live plot; drop it if the plot is behind
        try:
            sample_queue.put_nowait((t,
                                     rig["shoulder"].torque, predicted_shoulder_torque, gru.get_uncertainty_1(),
                                     rig["elbow"].torque, predicted_elbow_torque, gru.get_uncertainty_2()))
        except queue.Full:
            pass

        # command
        rig.set_positions(targets, kp=KP, kd=KD)

        # print a few times per second
        if tick % ticks_per_print == 0:
            print_status(t, rig, late_ticks, scores)

        # wait for the next tick
        tick += 1
        next_tick += period
        if time.perf_counter() > next_tick:
            late_ticks += 1
            next_tick = time.perf_counter()   # fell behind: carry on instead of catching up
        wait_until(next_tick)


def plot_errors(history):
    """Plot measured minus predicted torque for both joints."""
    figure, (shoulder_axis, elbow_axis) = plt.subplots(2, 1, sharex=True)

    shoulder_axis.plot(history["time"], history["shoulder_error"])
    shoulder_axis.axhline(0, color="gray", linewidth=0.5)
    shoulder_axis.set_ylabel("shoulder error (N*m)")

    elbow_axis.plot(history["time"], history["elbow_error"])
    elbow_axis.axhline(0, color="gray", linewidth=0.5)
    elbow_axis.set_ylabel("elbow error (N*m)")
    elbow_axis.set_xlabel("time (s)")

    figure.suptitle("Measured torque - predicted torque")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None, help="e.g. COM7")
    args = parser.parse_args()

    # Imported here, not at the top: collect_torque_data.py and the live plot process also
    # import this file, and neither of them needs PyTorch.
    from model import gru_prediction as gru
    print(f"using model {gru.load_model()}")

    history = {"time": [], "shoulder_error": [], "elbow_error": []}

    # start the live plot in its own process so drawing never slows the loop
    sample_queue = multiprocessing.Queue(maxsize=2000)
    plot_process = multiprocessing.Process(target=run_live_plot,
                                           args=(sample_queue, gru.TOUCH_SCORE_THRESHOLD), daemon=True)
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
