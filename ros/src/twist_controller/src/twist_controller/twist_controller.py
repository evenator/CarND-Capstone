"""Throttle, brake, and steering control."""
from math import atan
from rospy import loginfo, Publisher
from std_msgs.msg import Float32
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
        self.last_cmd_vel = 0
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

        self.debug_cmd_a = Publisher('~debug_cmd_acc', Float32, queue_size=2)
        self.debug_cmd_vel = Publisher('~debug_cmd_vel', Float32, queue_size=2)
        self.debug_cur_a = Publisher('~debug_cur_acc', Float32, queue_size=2)

    def control(self, cmd_lin_vel, cmd_ang_vel, cur_lin_vel, cur_ang_vel, delta_t, dbw_enabled):
        """
        Control update.

        This function should be called at the rate given in the constructor.
        Returns a tuple of (throttle, brake, steering)
        """
        # Steering
        if abs(cmd_lin_vel) > 0.1:
            curvature = cmd_ang_vel / cmd_lin_vel
        else:
            curvature = cmd_ang_vel / 0.1
        steering = self.calculate_steering(curvature)

        # Acceleration limiting (velocity ramp)
        min_vel = self.last_cmd_vel + self.decel_limit * delta_t
        max_vel = self.last_cmd_vel + self.accel_limit * delta_t
        cmd_lin_vel = max(min_vel, min(cmd_lin_vel, max_vel))
        self.last_cmd_vel = cmd_lin_vel

        # Acceleration Measurement Calculation
        vel_error = cmd_lin_vel - cur_lin_vel
        raw_acc = (cur_lin_vel - self.last_cur_lin_vel)/delta_t
        cur_acc = self.accel_filter.filt(raw_acc)
        self.last_cur_lin_vel = cur_lin_vel

        # If DBW not enabled, reset and return
        if not dbw_enabled:
            self.reset()
            return (0, 0, 0)

        # Speed PID to get commanded acceleration
        cmd_acc = self.speed_pid.step(vel_error, delta_t)

        # Acceleration limiting (commanded acceleration)
        cmd_acc = max(self.decel_limit, min(cmd_acc, self.accel_limit))

        # Throttle and brake calculations
        throttle_error = cmd_acc - cur_acc
        throttle = self.throttle_pid.step(throttle_error, delta_t)
        brake = self.calculate_brake(cmd_acc)

        if cmd_acc < 0:
            if brake < self.brake_deadband:
                brake = 0.
            throttle = 0.
        else:
            brake = 0.

        self.debug_cmd_vel.publish(Float32(data=cmd_lin_vel))
        self.debug_cmd_a.publish(Float32(data=cmd_acc))
        self.debug_cur_a.publish(Float32(data=cur_acc))
        return throttle, brake, steering

    def calculate_steering(self, curvature):
        # TODO: Remove this magic number
        return 0.8*atan(self.wheel_base * curvature) * self.steer_ratio

    def calculate_brake(self, cmd_acc):
        '''
        Calculate brake torque to achieve desired deceleration
        '''
        force = -cmd_acc * self.vehicle_mass
        return force * self.wheel_radius

    def reset(self):
        self.last_cur_lin_vel = 0
        self.speed_pid.reset()
        self.throttle_pid.reset()
