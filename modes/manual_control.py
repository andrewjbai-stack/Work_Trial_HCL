"""manual_control.py - window with sliders to position the two joints by hand.

    py modes/manual_control.py --port COM7

- Motors start DISABLED. Click Enable to turn them on; the sliders jump to the
  arm's current angles first, so nothing moves on enable.
- Drag a slider and the joint follows at no more than MAX_SPEED_DEG_PER_S.
- Esc (or the Disable button) turns the motors off. Closing the window does too.

Each joint's readout shows its touch score from the GRU, and the status line reports touches.
A second window plots measured torque against the GRU prediction live, while enabled.

All CAN traffic happens in one background thread. The window only reads and
writes shared values, so dragging a slider never delays a motor command.
"""

import argparse
import math
import multiprocessing
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # project folder, for dm_motor, model and the shared files
from dm_motor import Rig
from arm_loop import JOINT_LIMIT_DEG, JOINTS
from error_calculation import calculate_gravity_1, calculate_gravity_2
from live_plot import run_live_plot

# ---- settings -------------------------------------------------------------
LOOP_HZ = 200
KP = 20.0                       # stiffness while enabled
KD = 0.5                        # damping
MAX_SPEED_DEG_PER_S = 30        # joints never move faster than this
USE_GRAVITY_FEEDFORWARD = True  # add predicted gravity torque to each command
MAX_FEEDBACK_AGE = 0.1          # seconds without feedback before stopping
ACCELERATION_SMOOTHING = 0.1    # 0-1; lower = smoother but slower acceleration estimate
TOUCH_DISPLAY_S = 0.5           # keep showing a touch this long, so a brief one can be seen
# ---------------------------------------------------------------------------


class ArmController:
    """Runs the motor loop in a background thread."""

    def __init__(self, rig, gru, sample_queue=None):
        self.rig = rig
        self.gru = gru                        # the gru_prediction module
        self.normal_tx_gap = rig.bus.tx_gap_s
        self.sample_queue = sample_queue      # live plot; None = no plot
        self.names = rig.names()
        self.lock = threading.Lock()

        # written by the window
        self.slider_targets = {name: 0.0 for name in self.names}   # rad
        self.request = None          # "enable", "disable", "zero" or None

        # written by the loop, read by the window
        self.enabled = False
        self.commanded = {name: 0.0 for name in self.names}        # rad
        self.readings = {name: (math.nan, math.nan, math.nan) for name in self.names}
        self.status = "disabled"

        # for the GRU inputs
        self.start_time = time.perf_counter()
        self.previous_time = None
        self.previous_velocity = {name: 0.0 for name in self.names}
        self.acceleration = {name: 0.0 for name in self.names}

        # touch check, written by the loop and read by the window
        self.touch_scores = {name: math.nan for name in self.names}
        self.last_touch_time = -math.inf

        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)

    # -- called from the window -------------------------------------------
    def start(self):
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)

    def ask(self, request):
        with self.lock:
            self.request = request

    def set_target_deg(self, name, angle_deg):
        with self.lock:
            limited = max(-JOINT_LIMIT_DEG, min(JOINT_LIMIT_DEG, angle_deg))
            self.slider_targets[name] = math.radians(limited)

    # -- the loop (background thread) --------------------------------------
    def _loop(self):
        period = 1.0 / LOOP_HZ
        max_step = math.radians(MAX_SPEED_DEG_PER_S) * period
        next_tick = time.perf_counter()

        try:
            while self.running:
                with self.lock:
                    request, self.request = self.request, None

                if request == "enable" and not self.enabled:
                    self._enable()
                elif request == "disable" and self.enabled:
                    self._disable("disabled")
                elif request == "zero" and not self.enabled:
                    self._zero()

                with self.lock:
                    targets = dict(self.slider_targets)

                if self.enabled:
                    self._command(targets, max_step)
                    self.rig.spin_once(refresh=False)    # replies to our commands
                    self._check_safety()
                else:
                    self.rig.spin_once(refresh=True)     # poll for readings

                with self.lock:
                    for joint in self.rig.joints:
                        self.readings[joint.name] = (joint.position, joint.velocity, joint.torque)

                self._predict_and_plot()

                next_tick += period
                sleep_time = next_tick - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_tick = time.perf_counter()
        except Exception as problem:     # never leave motors on after a crash
            self._disable(f"STOPPED: {problem}")
        finally:
            if self.enabled:
                self._disable("disabled")

    def _predict_and_plot(self):
        """Predict each joint's torque with the GRU and send it to the plot."""
        if self.sample_queue is None or not ("shoulder" in self.names and "elbow" in self.names):
            return

        shoulder, elbow = self.rig["shoulder"], self.rig["elbow"]
        if any(math.isnan(value) for value in (shoulder.position, elbow.position,
                                               shoulder.velocity, elbow.velocity)):
            return

        # acceleration = change in velocity / change in time, lightly smoothed
        t = time.perf_counter() - self.start_time
        if self.previous_time is not None:
            dt = t - self.previous_time
            if dt > 0:
                for joint in (shoulder, elbow):
                    raw = (joint.velocity - self.previous_velocity[joint.name]) / dt
                    self.acceleration[joint.name] += ACCELERATION_SMOOTHING * (raw - self.acceleration[joint.name])
        self.previous_time = t
        for joint in (shoulder, elbow):
            self.previous_velocity[joint.name] = joint.velocity

        # The GRU only learned from powered motion, so it isn't fed while disabled. The
        # prediction is nan then, which shows as a gap in the plot.
        if self.enabled:
            self.gru.feed_live_data(shoulder.position, elbow.position,
                                    shoulder.velocity, elbow.velocity,
                                    self.acceleration["shoulder"], self.acceleration["elbow"])
        predicted_shoulder = self.gru.get_predicted_torque_1()
        predicted_elbow = self.gru.get_predicted_torque_2()

        scores = {"shoulder": self.gru.get_touch_score_1(shoulder.torque),
                  "elbow": self.gru.get_touch_score_2(elbow.torque)}
        with self.lock:
            self.touch_scores = scores
            if any(score > self.gru.TOUCH_SCORE_THRESHOLD for score in scores.values()):
                self.last_touch_time = t

        # drop the sample if the plot is behind, rather than slowing the loop
        try:
            self.sample_queue.put_nowait((t, shoulder.torque, predicted_shoulder, self.gru.get_uncertainty_1(),
                                          elbow.torque, predicted_elbow, self.gru.get_uncertainty_2()))
        except queue.Full:
            pass

    def _enable(self):
        positions = self.rig.positions()
        if any(math.isnan(q) for q in positions.values()):
            with self.lock:
                self.status = "no feedback yet - can't enable"
            return
        for name, angle in positions.items():
            if abs(math.degrees(angle)) > JOINT_LIMIT_DEG:
                with self.lock:
                    self.status = (f"{name} reads {math.degrees(angle):+.0f} deg, outside +-{JOINT_LIMIT_DEG}: "
                                   "hold it inside the limit and Enable again, or re-zero with the arm straight up")
                return
        with self.lock:
            self.slider_targets = dict(positions)      # start where the arm already is
            self.commanded = dict(positions)
            self.status = "enabled"
        self.rig.enable_all()

        # The SDK sleeps 3 ms after every frame by default, which would use up most of a 5 ms
        # tick. The GRU was trained at 200 Hz, so the loop needs to actually run at 200 Hz.
        self.rig.bus.tx_gap_s = 0.0
        self.gru.reset_memory()
        self.enabled = True

    def _disable(self, status):
        self.rig.disable_all()
        self.rig.bus.tx_gap_s = self.normal_tx_gap
        self.gru.reset_memory()
        self.enabled = False
        with self.lock:
            self.status = status

    def _zero(self):
        for joint in self.rig.joints:
            joint.motor.set_zero()
            time.sleep(0.02)
        time.sleep(0.05)
        for joint in self.rig.joints:
            joint.refresh()
            time.sleep(0.02)
        with self.lock:
            self.slider_targets = {name: 0.0 for name in self.names}
            self.status = "zeroed: this pose is now straight up, elbow straight"

    def _command(self, targets, max_step):
        # move each commanded angle toward its slider, at most max_step per tick
        for name in self.names:
            difference = targets[name] - self.commanded[name]
            step = max(-max_step, min(max_step, difference))
            self.commanded[name] += step

        feedforward = None
        if USE_GRAVITY_FEEDFORWARD and "shoulder" in self.names and "elbow" in self.names:
            shoulder = self.rig["shoulder"].position
            elbow = self.rig["elbow"].position
            feedforward = {"shoulder": calculate_gravity_1(shoulder, elbow),
                           "elbow": calculate_gravity_2(shoulder, elbow)}

        self.rig.set_positions(self.commanded, kp=KP, kd=KD, feedforward=feedforward)

    def _check_safety(self):
        for joint in self.rig.joints:
            status = joint.telemetry()
            if not status.ok:
                self._disable(f"STOPPED: {joint.name} fault ({status.status_text})")
                return
            if status.age_s > MAX_FEEDBACK_AGE:
                self._disable(f"STOPPED: {joint.name} no feedback for {status.age_s * 1000:.0f} ms")
                return


class ControlWindow:
    """The tkinter window: one slider and readout per joint, plus buttons."""

    def __init__(self, root, controller):
        self.root = root
        self.controller = controller
        self.sliders = {}
        self.readouts = {}
        self.ignore_slider_events = False

        root.title("Arm manual control")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", lambda event: controller.ask("disable"))

        for row, name in enumerate(controller.names):
            limit = JOINT_LIMIT_DEG
            frame = tk.LabelFrame(root, text=name, padx=10, pady=5)
            frame.grid(row=row, column=0, columnspan=4, sticky="ew", padx=10, pady=5)

            slider = tk.Scale(frame, from_=-limit, to=limit, resolution=0.5,
                              orient="horizontal", length=450, label="target (deg)",
                              command=lambda value, n=name: self.on_slider(n, value))
            slider.pack(fill="x")
            self.sliders[name] = slider

            readout = tk.Label(frame, font=("Consolas", 11), anchor="w", justify="left")
            readout.pack(fill="x")
            self.readouts[name] = readout

        button_row = len(controller.names)
        tk.Button(root, text="Enable", width=12,
                  command=lambda: controller.ask("enable")).grid(row=button_row, column=0, pady=10)
        tk.Button(root, text="Disable (Esc)", width=12, fg="red",
                  command=lambda: controller.ask("disable")).grid(row=button_row, column=1)
        tk.Button(root, text="Centre (0, 0)", width=12,
                  command=self.centre).grid(row=button_row, column=2)
        tk.Button(root, text="Zero (arm straight up)", width=20,
                  command=self.zero_here).grid(row=button_row, column=3)

        self.status_label = tk.Label(root, font=("Consolas", 11), anchor="w")
        self.status_label.grid(row=button_row + 1, column=0, columnspan=4, sticky="ew", padx=10, pady=(0, 10))

        self.refresh_display()

    def on_slider(self, name, value):
        if not self.ignore_slider_events:
            self.controller.set_target_deg(name, float(value))

    def centre(self):
        for name, slider in self.sliders.items():
            slider.set(0)
            self.controller.set_target_deg(name, 0)

    def zero_here(self):
        if self.controller.enabled:
            self.status_label.config(text="disable the motors before zeroing")
            return
        self.controller.ask("zero")

    def refresh_display(self):
        controller = self.controller
        with controller.lock:
            readings = dict(controller.readings)
            commanded = dict(controller.commanded)
            targets = dict(controller.slider_targets)
            status = controller.status
            enabled = controller.enabled
            scores = dict(controller.touch_scores)
            touch_age = time.perf_counter() - controller.start_time - controller.last_touch_time

        # keep sliders in sync when the loop moves them (enable, zero)
        self.ignore_slider_events = True
        for name, slider in self.sliders.items():
            target_deg = math.degrees(targets[name])
            if abs(slider.get() - target_deg) > 0.25:
                slider.set(round(target_deg * 2) / 2)
        self.ignore_slider_events = False

        for name, label in self.readouts.items():
            position, velocity, torque = readings[name]
            label.config(text=(f"measured {math.degrees(position):+7.1f} deg   "
                               f"commanded {math.degrees(commanded[name]):+7.1f} deg   "
                               f"velocity {math.degrees(velocity):+7.1f} deg/s   "
                               f"torque {torque:+.3f} N*m   "
                               f"score {scores[name]:4.1f}"))

        touched = enabled and touch_age < TOUCH_DISPLAY_S
        self.status_label.config(text=f"status: {status}" + ("   TOUCH" if touched else ""),
                                 fg="red" if touched or "STOPPED" in status else ("green" if enabled else "black"))
        self.root.after(100, self.refresh_display)

    def close(self):
        self.controller.stop()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None, help="e.g. COM7")
    args = parser.parse_args()

    # Imported here, not at the top, so the live plot process doesn't load PyTorch too.
    from model import gru_prediction as gru
    print(f"using model {gru.load_model()}")

    with Rig.open(port=args.port, joint_maps=JOINTS) as rig:   # leaving this block disables motors
        not_answering = rig.ensure_feedback()
        if not_answering:
            ids = [motor.can_id for motor in not_answering]
            print(f"no reply from motor id(s) {ids}")
            return

        # live plot of measured vs predicted torque, in its own process
        sample_queue = multiprocessing.Queue(maxsize=2000)
        plot_process = multiprocessing.Process(target=run_live_plot,
                                               args=(sample_queue, gru.TOUCH_SCORE_THRESHOLD), daemon=True)
        plot_process.start()

        controller = ArmController(rig, gru, sample_queue)
        controller.start()

        root = tk.Tk()
        ControlWindow(root, controller)
        try:
            root.mainloop()
        finally:
            controller.stop()
            rig.disable_all()
            print("motors disabled")
            plot_process.terminate()


if __name__ == "__main__":
    main()
