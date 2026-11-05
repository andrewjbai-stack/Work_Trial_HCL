"""live_plot.py - real-time torque plot, run in its own process.

The control loop puts one sample per tick into a queue:
    (time, shoulder_measured, shoulder_predicted, shoulder_uncertainty,
           elbow_measured, elbow_predicted, elbow_uncertainty)
This process empties the queue a few times per second and redraws, so drawing
never slows the control loop down.
"""

from collections import deque
import queue as queue_module

# ---- which lines to draw, on both joints' plots -----------------------------
SHOW_MEASURED = True
SHOW_PREDICTED = True
SHOW_ERROR = False          # measured - predicted
SHOW_UNCERTAINTY = False    # the model's uncertainty itself (one standard deviation)
SHOW_TOUCH_LIMITS = True   # predicted +- touch score * uncertainty: measured outside these is a touch
SHOW_TOUCH_BACKGROUND = True   # red background wherever measured is outside the touch limits
SHOW_PRESS_TEXT = True     # text saying which link is being pressed and which way
# -----------------------------------------------------------------------------

WINDOW_SECONDS = 10        # how much history to show
REDRAW_MS = 50             # how often to redraw (50 ms = 20 times per second)
MAX_SAMPLES = 200 * WINDOW_SECONDS
MIN_TOUCH_WIDTH_S = 0.03   # draw short touches at least this wide: one 5 ms tick is under a pixel wide
PRESS_TEXT_HOLD_S = 0.5    # keep showing a press this long, so a brief one can be read


def touch_rectangles(times, history):
    """One rectangle per stretch of time with measured outside the touch limits. x is in
    seconds and y runs from 0 to 1, the full height of the axis. Comparisons with nan are
    False, so nothing is marked while the model is warming up."""
    rectangles = []
    start = previous = None
    for t, measured, upper, lower in zip(times, history["measured"], history["upper limit"], history["lower limit"]):
        if measured > upper or measured < lower:
            if start is None:
                start = t
            previous = t
        elif start is not None:
            rectangles.append((start, previous))
            start = None
    if start is not None:
        rectangles.append((start, previous))

    shapes = []
    for first, last in rectangles:
        last = max(last, first + MIN_TOUCH_WIDTH_S)
        shapes.append([(first, 0), (last, 0), (last, 1), (first, 1)])
    return shapes


def describe_press(s_meas, s_pred, s_unc, e_meas, e_pred, e_unc, touch_score_threshold):
    """Which link is being pressed and which way, or None. A press on link 2 turns the elbow,
    while a press on link 1 can only turn the shoulder, because the elbow is beyond it.

    The motor fights the press, so its torque moves the opposite way from the push:
    measured above predicted means the arm is being pushed toward the negative direction."""
    shoulder_error = s_meas - s_pred
    elbow_error = e_meas - e_pred

    if abs(elbow_error) > touch_score_threshold * e_unc:
        return f"Link 2 pressed, pushing elbow toward {'-' if elbow_error > 0 else '+'}"
    if abs(shoulder_error) > touch_score_threshold * s_unc:
        return f"Link 1 pressed, pushing shoulder toward {'-' if shoulder_error > 0 else '+'}"
    return None


def make_lines(axis):
    """Draw the switched-on lines on one joint's axis. Fixed colours, so a line keeps its
    colour when others are switched off."""
    lines = {}
    if SHOW_MEASURED:
        lines["measured"] = axis.plot([], [], color="C0", label="measured")[0]
    if SHOW_PREDICTED:
        lines["predicted"] = axis.plot([], [], color="C1", label="predicted")[0]
    if SHOW_ERROR:
        lines["error"] = axis.plot([], [], color="C2", label="error", linestyle="--")[0]
    if SHOW_UNCERTAINTY:
        lines["uncertainty"] = axis.plot([], [], color="C3", label="uncertainty")[0]
    if SHOW_TOUCH_LIMITS:
        lines["upper limit"] = axis.plot([], [], color="C1", linewidth=0.7, linestyle=":", label="touch limits")[0]
        lines["lower limit"] = axis.plot([], [], color="C1", linewidth=0.7, linestyle=":")[0]
    return lines


def run_live_plot(sample_queue, touch_score_threshold):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import PolyCollection

    times = deque(maxlen=MAX_SAMPLES)
    names = ("measured", "predicted", "error", "uncertainty", "upper limit", "lower limit")
    shoulder_history = {name: deque(maxlen=MAX_SAMPLES) for name in names}
    elbow_history = {name: deque(maxlen=MAX_SAMPLES) for name in names}

    figure, (shoulder_axis, elbow_axis) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    shoulder_lines = make_lines(shoulder_axis)
    elbow_lines = make_lines(elbow_axis)

    shoulder_axis.set_ylabel("shoulder torque (N*m)")
    elbow_axis.set_ylabel("elbow torque (N*m)")
    elbow_axis.set_xlabel("time (s)")
    for axis, lines in ((shoulder_axis, shoulder_lines), (elbow_axis, elbow_lines)):
        axis.axhline(0, color="gray", linewidth=0.5)
        if lines:
            axis.legend(loc="upper left")
        axis.grid(alpha=0.3)
    figure.suptitle("Measured vs predicted torque (live)")
    press_label = figure.text(0.5, 0.925, "", ha="center", va="top", color="red", fontweight="bold", fontsize=12)

    # the red backgrounds: x in seconds, y as a fraction of the axis height
    def make_background(axis):
        background = PolyCollection([], facecolors="red", edgecolors="none", alpha=0.25, zorder=0,
                                    transform=axis.get_xaxis_transform())
        axis.add_collection(background, autolim=False)
        return background

    shoulder_background = make_background(shoulder_axis)
    elbow_background = make_background(elbow_axis)

    def remember(history, measured, predicted, uncertainty):
        margin = touch_score_threshold * uncertainty
        history["measured"].append(measured)
        history["predicted"].append(predicted)
        history["error"].append(measured - predicted)
        history["uncertainty"].append(uncertainty)
        history["upper limit"].append(predicted + margin)
        history["lower limit"].append(predicted - margin)

    stopped = False
    last_press = None
    last_press_time = 0.0

    def update(_frame):
        nonlocal stopped, last_press, last_press_time

        # 1. Take every sample that arrived since the last redraw.
        while not stopped:
            try:
                sample = sample_queue.get_nowait()
            except queue_module.Empty:
                break
            if sample is None:          # control loop finished
                stopped = True
                figure.suptitle("Measured vs predicted torque (stopped)")
                break
            t, s_meas, s_pred, s_unc, e_meas, e_pred, e_unc = sample
            times.append(t)
            remember(shoulder_history, s_meas, s_pred, s_unc)
            remember(elbow_history, e_meas, e_pred, e_unc)

            press = describe_press(s_meas, s_pred, s_unc, e_meas, e_pred, e_unc, touch_score_threshold)
            if press:
                last_press = press
                last_press_time = t

        if not times:
            return

        if SHOW_PRESS_TEXT:
            recent = last_press and times[-1] - last_press_time < PRESS_TEXT_HOLD_S
            press_label.set_text(last_press if recent else "")

        # 2. Redraw the lines.
        for name, line in shoulder_lines.items():
            line.set_data(times, shoulder_history[name])
        for name, line in elbow_lines.items():
            line.set_data(times, elbow_history[name])
        if SHOW_TOUCH_BACKGROUND:
            shoulder_background.set_verts(touch_rectangles(times, shoulder_history))
            elbow_background.set_verts(touch_rectangles(times, elbow_history))

        # 3. Scroll the time axis and rescale the torque axes.
        latest = times[-1]
        elbow_axis.set_xlim(max(0.0, latest - WINDOW_SECONDS), max(WINDOW_SECONDS, latest))
        for axis in (shoulder_axis, elbow_axis):
            axis.relim()
            axis.autoscale_view(scalex=False, scaley=True)

    animation = FuncAnimation(figure, update, interval=REDRAW_MS, cache_frame_data=False)  # noqa: F841

    try:
        plt.show()
    except KeyboardInterrupt:
        pass   # Ctrl+C in the terminal also reaches this process; just close quietly
