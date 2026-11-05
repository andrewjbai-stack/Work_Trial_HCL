"""collect_torque_data.py - drive both joints randomly and record training data.

    py modes/collect_torque_data.py --port COM7 --minutes 1 --scale 0.3     (gentle first try)
    py modes/collect_torque_data.py --port COM7 --minutes 30                (real recording)

Each joint moves on its own: random goals inside +-JOINT_LIMIT_DEG, random speeds,
random accelerations and random pauses. Normal moves are 5th-order polynomials, so
position, velocity and acceleration never jump. Mixed in are sharp reversals: runs at
constant speed where the commanded velocity flips instantly, like arm_loop.py's triangle wave.

Output goes in data/: a .csv with one row per tick and a .json describing
the run. Columns starting with cmd_ are what was commanded. They are there for
checking the run only and must NEVER be model inputs, because a model fed the target
just learns the motor's control law and then cannot notice a touch.

Ctrl+C once: finish the current move, go back to 0 deg, hold 3 s, then motors off.
Ctrl+C twice: motors off immediately (the arm will fall, so hold it).
"""

import argparse
import csv
import json
import math
import random
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # project folder, for dm_motor, model and the shared files
from dm_motor import Rig
from arm_loop import (ACCELERATION_SMOOTHING, JOINT_LIMIT_DEG, JOINTS, LOOP_HZ, check_safety,
                      wait_until, zero_all_motors)
from error_calculation import calculate_gravity_1, calculate_gravity_2

# ---- settings -------------------------------------------------------------
# Gains from manual_control.py, not arm_loop.py. With KP = 5 and no feedforward,
# gravity pulls a 60 deg target out to about 77-87 deg, past the +-60 deg limit.
# Keep these the same between recording sessions.
KP = 20.0
KD = 0.5
USE_GRAVITY_FEEDFORWARD = True

# Random motion. Each move picks its own limits between MIN and MAX (--scale lowers MAX).
MIN_SPEED_DEG_PER_S = 3
MAX_SPEED_DEG_PER_S = 90
MIN_ACCEL_DEG_PER_S2 = 30
MAX_ACCEL_DEG_PER_S2 = 300
MIN_MOVE_S = 0.5            # shorter moves would mean sharper changes in acceleration
MAX_MOVE_S = 5.0
STOP_CHANCE = 0.5           # chance a move ends at rest instead of flowing into the next
PAUSE_CHANCE = 0.7          # chance of a pause after a move that ends at rest
MIN_PAUSE_S = 0.1
MAX_PAUSE_S = 3.0

# Sharp reversals. Can start whenever a joint is at rest. Each flip is a torque kick of
# about KD * 2 * speed (0.8 N*m at 45 deg/s), so raise the top speed with care.
SHARP_REVERSAL_CHANCE = 0.9
MIN_REVERSAL_SPEED_DEG_PER_S = 10
MAX_REVERSAL_SPEED_DEG_PER_S = 45
MIN_REVERSAL_LEG_S = 0.2    # time at constant speed between flips
MAX_REVERSAL_LEG_S = 1.0
MIN_REVERSAL_FLIPS = 2
MAX_REVERSAL_FLIPS = 8

# Start and end
START_HOLD_S = 3.0          # hold still after enabling; let go of the arm during this
END_HOLD_S = 3.0            # hold at 0 deg before the motors turn off
HOME_SPEED_DEG_PER_S = 45   # speed limit for the move back to 0 deg

# Safety. Going past the position limit means something is wrong, so motors go off.
# Reaching the temperature limit ends the recording normally (back to 0 deg first).
MAX_OVERSHOOT_DEG = 15
MAX_TEMP_C = 70

OUTPUT_FOLDER = Path(__file__).resolve().parent.parent / "data"
PRINT_EVERY_S = 1.0
# ---------------------------------------------------------------------------


class Move:
    """Smooth path from (q0, v0) to (q1, v1) over T seconds, zero acceleration at both ends."""

    def __init__(self, q0, v0, q1, v1, T):
        self.q1 = q1
        self.v1 = v1
        self.T = T
        h = q1 - q0 - v0 * T
        k = v1 - v0
        self.c = (q0,
                  v0,
                  0.0,
                  (10 * h - 4 * k * T) / T ** 3,
                  (-15 * h + 7 * k * T) / T ** 4,
                  (6 * h - 3 * k * T) / T ** 5)

    def at(self, s):
        """Position, velocity and acceleration, s seconds into the move."""
        c0, c1, c2, c3, c4, c5 = self.c
        q = c0 + s * (c1 + s * (c2 + s * (c3 + s * (c4 + s * c5))))
        v = c1 + s * (2 * c2 + s * (3 * c3 + s * (4 * c4 + s * 5 * c5)))
        a = 2 * c2 + s * (6 * c3 + s * (12 * c4 + s * 20 * c5))
        return q, v, a

    def stays_within(self, q_limit, v_limit, a_limit):
        for i in range(31):
            q, v, a = self.at(self.T * i / 30)
            if abs(q) > q_limit or abs(v) > v_limit or abs(a) > a_limit:
                return False
        return True


def shortest_time(distance, v_change, speed, accel):
    """Rough minimum duration, using a smooth move's peak speed and acceleration."""
    return max(MIN_MOVE_S,
               1.875 * abs(distance) / speed,
               math.sqrt(5.77 * abs(distance) / accel),
               1.875 * abs(v_change) / accel)


class RandomJointMotion:
    """One joint's random path. Ask it where to be with at(t), for increasing t (rad units)."""

    def __init__(self, rng, limit, max_speed, max_accel):
        self.rng = rng
        self.limit = limit
        self.min_speed = min(math.radians(MIN_SPEED_DEG_PER_S), max_speed)
        self.max_speed = max_speed
        self.min_accel = min(math.radians(MIN_ACCEL_DEG_PER_S2), max_accel)
        self.max_accel = max_accel
        self.move = None
        self.move_start = 0.0
        self.planned_moves = []    # the rest of a sharp reversal sequence
        self.going_home = False
        self.is_home = False       # at 0 deg and stopped

    def start(self, t, position):
        """Begin at rest at `position`, holding still until the first move at time t."""
        self.move = Pause(position, 1e-3)
        self.move_start = t - 1e-3

    def go_home(self):
        """Head to 0 deg as soon as the current move finishes."""
        self.going_home = True

    def at(self, t):
        while t - self.move_start >= self.move.T:
            self.move_start += self.move.T
            self.move = self._next_move()
        return self.move.at(t - self.move_start)

    def _next_move(self):
        q0, v0 = self.move.q1, self.move.v1

        if self.going_home:
            if q0 == 0 and v0 == 0:
                self.is_home = True
                return Pause(0.0, 0.2)
            return self._brake(q0, v0) if v0 != 0 else self._move_home(q0)

        if self.planned_moves:
            return self.planned_moves.pop(0)

        # after a stop, maybe stay there for a while
        if v0 == 0 and not isinstance(self.move, Pause) and self.rng.random() < PAUSE_CHANCE:
            return Pause(q0, self.rng.uniform(MIN_PAUSE_S, MAX_PAUSE_S))

        if v0 == 0 and self.rng.random() < SHARP_REVERSAL_CHANCE:
            moves = self._plan_sharp_reversals(q0)
            if moves:
                self.planned_moves = moves[1:]
                return moves[0]

        for _ in range(100):
            # log-uniform, so slow moves get as much time in the data as fast ones
            speed = math.exp(self.rng.uniform(math.log(self.min_speed), math.log(self.max_speed)))
            accel = math.exp(self.rng.uniform(math.log(self.min_accel), math.log(self.max_accel)))
            speed_allowed = max(speed, abs(v0))

            # pick a goal, but only as far as this speed can get to in one move
            goal = self.rng.uniform(-self.limit, self.limit)
            reach = speed * MAX_MOVE_S / 1.875
            distance = max(-reach, min(reach, goal - q0))
            q1 = q0 + distance

            # keep moving through the goal only if there is room left to brake before the limit
            v1 = 0.0
            if self.rng.random() > STOP_CHANCE:
                candidate = math.copysign(speed * self.rng.uniform(0.3, 0.9), distance)
                room = self.limit - q1 * math.copysign(1.0, candidate)
                if self._braking_distance(candidate) < room:
                    v1 = candidate

            T = shortest_time(distance, v1 - v0, speed_allowed, accel) * self.rng.uniform(1.0, 1.4)
            for stretch in (1.0, 1.3, 1.7, 2.2, 3.0):
                if T * stretch > MAX_MOVE_S:
                    break
                move = Move(q0, v0, q1, v1, T * stretch)
                if move.stays_within(self.limit, speed_allowed, self.max_accel):
                    return move

        return self._brake(q0, v0)   # no random move fitted (rare)

    def _plan_sharp_reversals(self, q0):
        """Speed up smoothly from rest, then run back and forth, flipping velocity instantly.
        Every flip point leaves room to brake, so stopping is always possible. Returns [] if
        the sequence doesn't fit here."""
        top_speed = min(math.radians(MAX_REVERSAL_SPEED_DEG_PER_S), self.max_speed)
        low_speed = min(math.radians(MIN_REVERSAL_SPEED_DEG_PER_S), top_speed)
        speed = math.exp(self.rng.uniform(math.log(low_speed), math.log(top_speed)))
        room = self.limit - self._braking_distance(speed)

        velocity = self.rng.choice((-speed, speed))
        T = self._braking_time(speed)
        q = q0 + velocity * T / 2
        if abs(q) >= room:
            velocity = -velocity
            q = q0 + velocity * T / 2
            if abs(q) >= room:
                return []
        moves = [Move(q0, 0.0, q, velocity, T)]

        flips = self.rng.randint(MIN_REVERSAL_FLIPS, MAX_REVERSAL_FLIPS)
        for leg in range(flips + 1):
            T = self.rng.uniform(MIN_REVERSAL_LEG_S, MAX_REVERSAL_LEG_S)
            end = max(-room, min(room, q + velocity * T))
            T = abs(end - q) / speed
            if T < MIN_REVERSAL_LEG_S:     # too close to the limit: leave it early
                break
            moves.append(Move(q, velocity, end, velocity, T))
            q = end
            velocity = -velocity
        return moves if len(moves) > 2 else []

    def _braking_time(self, v):
        return max(MIN_MOVE_S, 1.875 * abs(v) / self.max_accel)

    def _braking_distance(self, v):
        return abs(v) * self._braking_time(v) / 2

    def _brake(self, q0, v0):
        """Come to rest as quickly as allowed. Moves only keep a speed they can brake from."""
        T = self._braking_time(v0)
        return Move(q0, v0, q0 + v0 * T / 2, 0.0, T)

    def _move_home(self, q0):
        """From rest to 0 deg."""
        speed = min(math.radians(HOME_SPEED_DEG_PER_S), self.max_speed)
        return Move(q0, 0.0, 0.0, 0.0, shortest_time(q0, 0.0, speed, self.max_accel))


class Pause(Move):
    def __init__(self, position, duration):
        super().__init__(position, 0.0, position, 0.0, duration)


def column_names(names):
    columns = ["t_s", "phase"]
    for name in names:
        columns += [f"{name}_q_rad", f"{name}_dq_rad_s", f"{name}_ddq_rad_s2", f"{name}_tau_nm",
                    f"{name}_mos_temp_c", f"{name}_rotor_temp_c", f"{name}_feedback_age_ms"]
    for name in names:
        columns += [f"cmd_{name}_q_rad", f"cmd_{name}_dq_rad_s", f"cmd_{name}_tau_ff_nm"]
    return columns


def describe_run(args, seed, limit_deg, max_speed, max_accel, columns):
    return {
        "started": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "minutes_requested": args.minutes,
        "scale": args.scale,
        "convention": "shoulder 0 = link 1 straight up, elbow 0 = straight, same rotation sense for +",
        "joints": [{"name": j.name, "can_id": j.can_id, "sign": j.sign} for j in JOINTS],
        "loop_hz": LOOP_HZ,
        "kp": KP,
        "kd": KD,
        "gravity_feedforward": USE_GRAVITY_FEEDFORWARD,
        "joint_limit_deg": limit_deg,
        "max_speed_deg_per_s": math.degrees(max_speed),
        "max_accel_deg_per_s2": math.degrees(max_accel),
        "acceleration": f"measured dq differentiated live, smoothing {ACCELERATION_SMOOTHING}",
        "sharp_reversals": {"chance": SHARP_REVERSAL_CHANCE,
                            "max_speed_deg_per_s": MAX_REVERSAL_SPEED_DEG_PER_S},
        "columns": columns,
        "not_model_inputs": [c for c in columns if c.startswith("cmd_") or c == "phase"],
        "training_rows": "use phase == motion; settle, return and hold are the start and end of the run",
    }


def gravity_feedforward(targets):
    """Torque to hold the commanded pose, so tracking error stays small near the limits."""
    if not USE_GRAVITY_FEEDFORWARD:
        return {"shoulder": 0.0, "elbow": 0.0}
    return {"shoulder": calculate_gravity_1(targets["shoulder"], targets["elbow"]),
            "elbow": calculate_gravity_2(targets["shoulder"], targets["elbow"])}


def run(rig, args, summary):
    not_answering = rig.ensure_feedback()
    if not_answering:
        raise RuntimeError(f"no reply from motor id(s) {[m.can_id for m in not_answering]}")

    zero_all_motors(rig)

    # first Ctrl+C asks for a clean finish, second one drops out immediately
    stop_asked = []

    def on_ctrl_c(signum, frame):
        if stop_asked:
            raise KeyboardInterrupt
        stop_asked.append(True)

    signal.signal(signal.SIGINT, on_ctrl_c)

    seed = args.seed if args.seed is not None else random.randrange(2 ** 31)
    limit = math.radians(JOINT_LIMIT_DEG)
    max_speed = math.radians(MAX_SPEED_DEG_PER_S) * args.scale
    max_accel = math.radians(MAX_ACCEL_DEG_PER_S2) * args.scale
    trip_angle = limit + math.radians(MAX_OVERSHOOT_DEG)

    names = rig.names()
    columns = column_names(names)
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    stem = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    csv_path = OUTPUT_FOLDER / f"{stem}.csv"
    json_path = OUTPUT_FOLDER / f"{stem}.json"
    info = describe_run(args, seed, JOINT_LIMIT_DEG, max_speed, max_accel, columns)
    json_path.write_text(json.dumps(info, indent=2))
    print(f"recording to {csv_path}")

    rig.enable_all()
    rig.spin_once(refresh=False)

    # The SDK sleeps 3 ms after every frame by default, which would use up most of a 5 ms
    # tick. Both frames now go out together and the wait for the next tick is the gap.
    rig.bus.tx_gap_s = 0.0

    start_pose = rig.positions()
    motions = {}
    for index, name in enumerate(names):
        motions[name] = RandomJointMotion(random.Random(f"{seed}-{index}"), limit, max_speed, max_accel)
        motions[name].start(START_HOLD_S, start_pose[name])
    print(f"holding for {START_HOLD_S:.0f} s: LET GO OF THE ARM and stand clear")

    period = 1.0 / LOOP_HZ
    duration = args.minutes * 60.0
    stop_reason = None
    phase = "settle"
    hold_until = None
    tick = 0
    late_ticks = 0
    next_print = 0.0
    start_time = time.perf_counter()
    next_tick = start_time
    previous_time = None
    previous_velocity = {}
    acceleration = {name: 0.0 for name in names}

    # big buffer: writes reach the disk about every 15 s, so they rarely delay a tick
    with open(csv_path, "w", newline="", buffering=2 ** 20) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(columns)

        while True:
            # read
            rig.spin_once(refresh=False)
            t = time.perf_counter() - start_time
            check_safety(rig)
            status = {joint.name: joint.telemetry() for joint in rig.joints}

            for name in names:
                if abs(status[name].position) > trip_angle:
                    raise RuntimeError(f"{name} reached {math.degrees(status[name].position):+.1f} deg")

            # acceleration = change in measured velocity / change in time, lightly smoothed.
            # Worked out live, because the touch detector will only have past readings too.
            if previous_time is not None and t > previous_time:
                for name in names:
                    raw = (status[name].velocity - previous_velocity[name]) / (t - previous_time)
                    acceleration[name] += ACCELERATION_SMOOTHING * (raw - acceleration[name])
            previous_time = t
            previous_velocity = {name: status[name].velocity for name in names}

            # decide whether it is time to finish
            if stop_reason is None:
                hottest = max(max(s.mos_temp_c, s.rotor_temp_c) for s in status.values())
                if stop_asked:
                    stop_reason = "ctrl-c"
                elif t >= duration + START_HOLD_S:
                    stop_reason = "time up"
                elif hottest >= MAX_TEMP_C:
                    stop_reason = f"motor reached {hottest:.0f} C"
                if stop_reason:
                    print(f"\nfinishing ({stop_reason}): returning to 0 deg")
                    for motion in motions.values():
                        motion.go_home()

            # decide: where each joint should be now
            q_target, v_target = {}, {}
            for name in names:
                q_target[name], v_target[name], _ = motions[name].at(t)
            feedforward = gravity_feedforward(q_target)

            if phase == "settle" and t >= START_HOLD_S:
                phase = "motion"
                print("moving")
            elif phase == "motion" and stop_reason:
                phase = "return"
            elif phase == "return" and all(motion.is_home for motion in motions.values()):
                phase = "hold"
                hold_until = t + END_HOLD_S
                print(f"at 0 deg: HOLD THE ARM, motors turn off in {END_HOLD_S:.0f} s")
            elif phase == "hold" and t >= hold_until:
                break

            # record
            row = [f"{t:.5f}", phase]
            for name in names:
                s = status[name]
                row += [f"{s.position:.5f}", f"{s.velocity:.4f}", f"{acceleration[name]:.3f}", f"{s.torque:.4f}",
                        f"{s.mos_temp_c:.0f}", f"{s.rotor_temp_c:.0f}", f"{s.age_s * 1000:.1f}"]
            for name in names:
                row += [f"{q_target[name]:.5f}", f"{v_target[name]:.4f}", f"{feedforward[name]:.4f}"]
            writer.writerow(row)

            # command
            for name in names:
                rig[name].set_position(q_target[name], kp=KP, kd=KD,
                                       velocity=v_target[name], feedforward=feedforward[name])

            if t >= next_print:
                next_print = t + PRINT_EVERY_S
                line = f"t={t:7.1f}s {phase:7s}"
                for name in names:
                    s = status[name]
                    line += (f"  {name}: {math.degrees(s.position):+6.1f} deg"
                             f" {s.torque:+.2f} Nm {s.rotor_temp_c:.0f} C")
                print(line + f"  late: {late_ticks}")

            # wait for the next tick
            tick += 1
            next_tick += period
            now = time.perf_counter()
            if now > next_tick:
                late_ticks += 1
                if now > next_tick + period:      # far behind: skip ahead instead of rushing
                    next_tick = now
            wait_until(next_tick)

            summary.update(info=info, json_path=json_path, ticks=tick, late_ticks=late_ticks,
                           seconds=t, stop_reason=stop_reason)


def main():
    parser = argparse.ArgumentParser(description="Record random-motion torque data.")
    parser.add_argument("--port", default=None, help="e.g. COM7")
    parser.add_argument("--minutes", type=float, default=30.0, help="length of the random motion")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="0-1, lowers the maximum speed and acceleration; try 0.3 first")
    parser.add_argument("--seed", type=int, default=None, help="repeat a run's path; random if left out")
    args = parser.parse_args()
    if not 0 < args.scale <= 1:
        parser.error("--scale must be above 0 and at most 1")

    summary = {}
    with Rig.open(port=args.port, joint_maps=JOINTS) as rig:   # leaving this block disables motors
        try:
            run(rig, args, summary)
        except KeyboardInterrupt:
            print("\nstopped immediately by user")
            summary["stop_reason"] = "second ctrl-c"
        except RuntimeError as problem:
            print(f"\nSTOP: {problem}")
            summary["stop_reason"] = f"fault: {problem}"
        finally:
            rig.disable_all()
            print("motors disabled")
            summary["link_counters"] = rig.bus.counters()

    if "info" in summary:
        achieved = summary["ticks"] / summary["seconds"] if summary["seconds"] > 0 else 0.0
        summary["info"].update(ticks=summary["ticks"], late_ticks=summary["late_ticks"],
                               seconds=summary["seconds"], achieved_hz=achieved,
                               stop_reason=summary["stop_reason"],
                               link_counters=summary["link_counters"])
        summary["json_path"].write_text(json.dumps(summary["info"], indent=2))
        print(f"{summary['ticks']} rows at {achieved:.1f} Hz, {summary['late_ticks']} late ticks")


if __name__ == "__main__":
    main()
