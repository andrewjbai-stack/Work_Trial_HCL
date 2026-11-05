import argparse
import math
import time

LENGTH_1 = 0.394
LENGTH_2 = 0.16

CENTER_OF_MASS_1 = 0.1945
CENTER_OF_MASS_2 = 0.03

LINK_MASS_1 = 0.1
LINK_MASS_2 = 0.03

JOINT_MASS_1 = 0.3
JOINT_MASS_2 = 0.3

GRAVITY = 9.81

LINK_INERTIA_1 = 0.05538
LINK_INERTIA_2 = 0.000373

VISCOUS_FRICTION_1 = 0.03
VISCOUS_FRICTION_2 = 0.03

COULOMB_FRICTION_1 = 0.18  
COULOMB_FRICTION_2 = 0.15  

FRICTION_SMOOTHING_SPEED = 0.05


# Angle convention:
#   theta1 = 0 is link 1 straight up, +-90 deg is horizontal (measured from vertical).
#   theta2 = 0 is the elbow fully straight, + turns the same way as + theta1.
# Gravity pulls a tilted arm further over, so the torque needed to hold it opposes
# the tilt: it has the opposite sign to the horizontal offset of each mass.
def calculate_gravity_1(theta1, theta2):
    link_2_absolute_angle = theta1 + theta2

    link_1_horizontal = CENTER_OF_MASS_1 * math.sin(theta1)
    elbow_horizontal = LENGTH_1 * math.sin(theta1)
    link_2_horizontal = (LENGTH_1 * math.sin(theta1)
                         + CENTER_OF_MASS_2 * math.sin(link_2_absolute_angle))

    link_1_torque = LINK_MASS_1 * GRAVITY * link_1_horizontal
    elbow_motor_torque = JOINT_MASS_2 * GRAVITY * elbow_horizontal
    link_2_torque = LINK_MASS_2 * GRAVITY * link_2_horizontal

    return -(link_1_torque + elbow_motor_torque + link_2_torque)


def calculate_gravity_2(theta1, theta2):
    link_2_absolute_angle = theta1 + theta2

    link_2_horizontal = CENTER_OF_MASS_2 * math.sin(link_2_absolute_angle)

    return -LINK_MASS_2 * GRAVITY * link_2_horizontal


def calculate_inertia_1(theta2, theta1_acceleration, theta2_acceleration):
    cos_theta2 = math.cos(theta2)

    link_1_inertia = LINK_INERTIA_1 + LINK_MASS_1 * CENTER_OF_MASS_1 ** 2

    elbow_motor_inertia = JOINT_MASS_2 * LENGTH_1 ** 2

    r_squared = (LENGTH_1 ** 2 + CENTER_OF_MASS_2 ** 2
                 + 2 * LENGTH_1 * CENTER_OF_MASS_2 * cos_theta2)
    link_2_inertia = LINK_INERTIA_2 + LINK_MASS_2 * r_squared

    shoulder_inertia = link_1_inertia + elbow_motor_inertia + link_2_inertia

    elbow_coupling = (LINK_INERTIA_2
                      + LINK_MASS_2 * CENTER_OF_MASS_2 ** 2
                      + LINK_MASS_2 * LENGTH_1 * CENTER_OF_MASS_2 * cos_theta2)

    return (shoulder_inertia * theta1_acceleration
            + elbow_coupling * theta2_acceleration)


def calculate_inertia_2(theta2, theta1_acceleration, theta2_acceleration):
    cos_theta2 = math.cos(theta2)

    link_2_inertia_about_elbow = (LINK_INERTIA_2
                                  + LINK_MASS_2 * CENTER_OF_MASS_2 ** 2)
    link_2_angular_acceleration = theta1_acceleration + theta2_acceleration
    spinning_torque = link_2_inertia_about_elbow * link_2_angular_acceleration

    dragging_torque = (LINK_MASS_2 * LENGTH_1 * CENTER_OF_MASS_2
                       * cos_theta2 * theta1_acceleration)

    return spinning_torque + dragging_torque


def calculate_coriolis_coefficient(theta2):
    return LINK_MASS_2 * LENGTH_1 * CENTER_OF_MASS_2 * math.sin(theta2)


def calculate_coriolis_1(theta2, theta1_velocity, theta2_velocity):
    h = calculate_coriolis_coefficient(theta2)

    skater_torque = -2 * h * theta1_velocity * theta2_velocity
    elbow_swing_torque = -h * theta2_velocity ** 2

    return skater_torque + elbow_swing_torque


def calculate_coriolis_2(theta2, theta1_velocity):
    h = calculate_coriolis_coefficient(theta2)

    return h * theta1_velocity ** 2


def smooth_sign(velocity):
    return math.tanh(velocity / FRICTION_SMOOTHING_SPEED)


def calculate_friction_1(theta1_velocity):
    viscous = VISCOUS_FRICTION_1 * theta1_velocity
    coulomb = COULOMB_FRICTION_1 * smooth_sign(theta1_velocity)
    return viscous + coulomb


def calculate_friction_2(theta2_velocity):
    viscous = VISCOUS_FRICTION_2 * theta2_velocity
    coulomb = COULOMB_FRICTION_2 * smooth_sign(theta2_velocity)
    return viscous + coulomb


def calculate_predicted_torque_1(theta1, theta2,
                                 theta1_velocity, theta2_velocity,
                                 theta1_acceleration, theta2_acceleration):
    inertia = calculate_inertia_1(theta2, theta1_acceleration, theta2_acceleration)
    coriolis = calculate_coriolis_1(theta2, theta1_velocity, theta2_velocity)
    gravity = calculate_gravity_1(theta1, theta2)
    friction = calculate_friction_1(theta1_velocity)
    return inertia + coriolis + gravity + friction


def calculate_predicted_torque_2(theta1, theta2,
                                 theta1_velocity, theta2_velocity,
                                 theta1_acceleration, theta2_acceleration):
    inertia = calculate_inertia_2(theta2, theta1_acceleration, theta2_acceleration)
    coriolis = calculate_coriolis_2(theta2, theta1_velocity)
    gravity = calculate_gravity_2(theta1, theta2)
    friction = calculate_friction_2(theta2_velocity)
    return inertia + coriolis + gravity + friction
