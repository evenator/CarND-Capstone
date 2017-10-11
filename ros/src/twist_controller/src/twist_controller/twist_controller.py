"""Throttle, brake, and steering control."""
from math import atan
from rospy import loginfo
from .lowpass import LowPassFilter
from .pid import PID

class Controller(object):
    """
    Class to calculate throttle, bake, and steering control.

    Uses PID controllers for steering and throttle.
    """

    def __init__(self, kp_vel, kp_throttle, ki_throttle, min_speed,
                 vehicle_mass, fuel_capacity, brake_deadband,
                 decel_limit, accel_limit,
                 wheel_radius, wheel_base,
                 steer_ratio, max_lat_accel, max_steer_angle, rate):
        """
        Constructor.

        Set up gains and initialize internal state; initialize PID controllers.
        """
        self.last_cur_lin_vel = 0
        self.accel_limit = accel_limit
        self.decel_limit = decel_limit
        self.vehicle_mass = vehicle_mass
        self.wheel_radius = wheel_radius
        self.brake_deadband = brake_deadband
        self.steer_ratio = steer_ratio
        self.wheel_base = wheel_base
        self.speed_pid = PID(kp_vel, 0.0, 0.0, self.decel_limit, self.accel_limit)
        self.accel_filter = LowPassFilter(0.2, 1./rate)
        self.throttle_pid = PID(kp_throttle, ki_throttle, 0.0, 0, 1.)

    def control(self, cmd_lin_vel, cmd_ang_vel, cur_lin_vel, cur_ang_vel, delta_t, dbw_enabled):
        """
        Control update.

        This function should be called at the rate given in the constructor.
        Returns a tuple of (throttle, brake, steering)
        """

        vel_error = cmd_lin_vel - cur_lin_vel
        raw_acc = cur_lin_vel - self.last_cur_lin_vel
        cur_acc = self.accel_filter.filt(raw_acc)
        self.last_cur_lin_vel = cur_lin_vel

        if not dbw_enabled:
            self.reset()
            return (0, 0, 0)

        if delta_t != 0.0:
            steer_error = (cmd_ang_vel - cur_ang_vel)/delta_t
        else:
            steer_error = 0.0

        if(abs(cmd_lin_vel)<1.):
            steering = 0
        else:
            steering = 0.8*atan(self.wheel_base * cmd_ang_vel / cmd_lin_vel) * self.steer_ratio

        cmd_acc = self.speed_pid.step(vel_error, delta_t)

        if cmd_acc > self.accel_limit:
            cmd_acc = self.accel_limit
        elif cmd_acc < self.decel_limit:
            cmd_acc = self.decel_limit

        throttle_error = (cmd_acc - cur_acc)

        brake = -cmd_acc*self.vehicle_mass*self.wheel_radius
        throttle = self.throttle_pid.step(throttle_error, delta_t)

        if cmd_acc < 0:
            if brake < self.brake_deadband:
                brake = 0.
            throttle = 0.
        else:
            brake = 0.

        return throttle, brake, steering

    def reset(self):
        self.last_cur_lin_vel = 0
        self.speed_pid.reset()
        self.throttle_pid.reset()
