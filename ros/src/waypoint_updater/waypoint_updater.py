#!/usr/bin/env python
"""
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.
"""

from geometry_msgs.msg import PoseStamped
import math
import rospy
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint, TrafficLight, TrafficLightArray


def get_waypoint_velocity(waypoint):
    """Get the velocity from a given waypoint."""
    return waypoint.twist.twist.linear.x


def set_waypoint_velocity(waypoints, index, velocity):
    """Set the velocity of the waypoint at the given index."""
    waypoints[index].twist.twist.linear.x = velocity


def distance(waypoints, index_1, index_2):
    """Calculate the distance between two waypoints using a piece-wise linear function."""
    dist = 0
    dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)
    for i in range(index_1, index_2+1):
        dist += dl(waypoints[index_1].pose.pose.position, waypoints[i].pose.pose.position)
        index_1 = i
    return dist

def position_dist(pos1, pos2):
    return math.sqrt( (pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2 + (pos1.z - pos2.z)**2 )


class WaypointUpdater(object):
    """A waypoint updater node implemented as a Python class."""
    RUNNING = 0
    STOPPING = 1
    STOPPED = 2
    ACCELERATING = 3

    def __init__(self):
        """
        Constructor.

        - Subscribes to current_pose and base_waypoints
        - Gets the lookahead parameter ~lookahead_wps
        - Advertises to final_waypoints
        - Sets up class variables
        """
        self.pose_sub = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        self.base_wp_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        self.traffic_lights_sub = rospy.Subscriber('/vehicle/traffic_lights',
                                                   TrafficLightArray,
                                                   self.gt_traffic_cb)
        self.traffic_wp_sub = rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.lookahead_wps = rospy.get_param('~lookahead_wps', 200)
        self.max_decel = abs(rospy.get_param('~max_deceleration', 1.0))
        self.use_ground_truth = rospy.get_param('~use_ground_truth', False)
        self.dist_threshold = rospy.get_param('~dist_threshold', 2.0)
        self.stopping_dist = rospy.get_param('~stopping_dist', 4.0)
        self.nb_stopping_wp = rospy.get_param('~nb_stopping_wp', 0)
        self.cruise_velocity = rospy.get_param('~cruise_velocity', 4.47)
        self.update_period = rospy.Duration(rospy.get_param('~update_period', 0.2))

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.lane = Lane()
        self.last_lane = Lane()

        self.traffic_lights = None

        self.state = WaypointUpdater.ACCELERATING

        self.traffic_light_wp_index = -1

        self.commanded_velocity = 0

        self.currpose = None
        self.curr_waypoints = None
        self.start_idx = 0

        self.last_time = rospy.Time(0)

    def get_state_string(self):
        return {WaypointUpdater.RUNNING: "Running",
                WaypointUpdater.STOPPING: "Stopping",
                WaypointUpdater.STOPPED: "Stopped",
                WaypointUpdater.ACCELERATING: "Accelerating"}[self.state]

    def pose_cb(self, msg):
        """Callback for the curr_pose just stores it for later use."""
        self.currpose = msg.pose.position
        self.update_lane()

    def waypoints_cb(self, lanemsg):
        """
        Callback for base_waypoints.

        Finds the waypoint closes to the current pose, then publishes the next lookahead_wps
        waypoints after that one to final_waypoints.
        """
        self.curr_waypoints = lanemsg.waypoints
       
    def update_lane(self):
        if self.curr_waypoints is None:
            return

        now = rospy.Time.now()
        if (now - self.last_time) < self.update_period:
            return

        self.last_time = now

        self.lane = Lane()
        self.lane.header.frame_id = '/world'
        self.lane.header.stamp = rospy.Time(0)
        mindist = 1000000
        start_idx = 0

        for i in range(len(self.curr_waypoints)):
            a = self.curr_waypoints[i].pose.pose.position
            b = self.currpose
            dist = math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)
            # if dist < mindist and (a.x > b.x):
            if dist < mindist:
                start_idx = i
                mindist = dist
            if start_idx == (len(self.curr_waypoints) - 1):
                start_idx = 0

        idx = 0
        reset = 0
        # Collecting the waypoints ahead of the car.
        # Wrap around when we reach the end.
        for i in range(self.lookahead_wps):
            idx = (start_idx + i) % len(self.curr_waypoints)
            self.lane.waypoints.append(self.curr_waypoints[idx])
        rospy.loginfo("Current waypoint: %d", start_idx)

        
        if self.use_ground_truth:
            self.get_traffic_from_gt()

        self.step()

        self.last_pose = self.currpose
        self.last_lane = self.lane
        self.start_idx = start_idx
        self.final_waypoints_pub.publish(self.lane)

    def get_traffic_from_gt(self):
        self.traffic_light_wp_index = -1
        # Check if there is a traffic light on the lookahead path
        for light in self.traffic_lights:
            if(light.state is not TrafficLight.RED):
                continue
            index = self.closest_index(light)
            if index >= self.nb_stopping_wp-5:
                self.traffic_light_wp_index = index-self.nb_stopping_wp
                return

    def step(self):
        if(len(self.last_lane.waypoints)>0):
            self.lane = self.keep_last(self.lane, self.last_lane)
            self.commanded_velocity = get_waypoint_velocity(self.lane.waypoints[0])
        else:
            self.lane = self.keep_speed(self.lane, self.cruise_velocity)
            self.commanded_velocity = self.cruise_velocity

        if(self.use_ground_truth):
            stop_index = self.traffic_light_wp_index
        else:
            stop_index = self.traffic_light_wp_index - self.start_idx
            if stop_index < 0:
                stop_index += len(self.curr_waypoints)
        
        rospy.loginfo("Traffic light index: %d Stop index: %d",
                      self.traffic_light_wp_index,
                      stop_index)
        if(self.state == WaypointUpdater.RUNNING):
            if self.traffic_light_wp_index >= 0:
                if stop_index < len(self.lane.waypoints):
                    # Red light within lane
                    # TRANSITION TO STOPPING
                    self.state = self.STOPPING
                    self.lane = self.braking(self.lane, stop_index, self.cruise_velocity)
        elif(self.state == WaypointUpdater.STOPPING):
            if self.traffic_light_wp_index < 0 or stop_index >= len(self.lane.waypoints):
                # Green/yellow light or next light past end of lane
                # TRANSITION TO ACCELERATING
                self.state = self.ACCELERATING
                self.lane = self.keep_speed(self.lane, self.cruise_velocity)
            elif(self.commanded_velocity<0.5):
                # TRANSITION TO STOPPED
                self.state = self.STOPPED
                self.lane = self.keep_speed(self.lane, 0)
        elif(self.state == WaypointUpdater.STOPPED):
            if self.traffic_light_wp_index < 0 or stop_index >= len(self.lane.waypoints):
                # Green/yellow light or next light last end of lane
                # TRANSITION TO ACCELERATING
                self.state = WaypointUpdater.ACCELERATING
                self.lane = self.keep_speed(self.lane, self.cruise_velocity)
        elif(self.state == WaypointUpdater.ACCELERATING):
            if(self.commanded_velocity>self.cruise_velocity-0.5):
                # At speed
                # TRANSITION TO RUNNING
                self.state = self.RUNNING
                self.lane = self.keep_speed(self.lane, self.cruise_velocity) 
        rospy.loginfo("State is %s", self.get_state_string())

    def keep_last(self, base_lane, last_lane):

        min_dist, curr_index = self.find_index(base_lane.waypoints[0].pose.pose.position, last_lane.waypoints)

        new_waypoint_list = last_lane.waypoints[curr_index:]

        last_vel = get_waypoint_velocity(last_lane.waypoints[-1])
        for i in range(len(new_waypoint_list), len(base_lane.waypoints)):
            set_waypoint_velocity(base_lane.waypoints, i, last_vel)
            new_waypoint_list.append(base_lane.waypoints[i])

        base_lane.waypoints = new_waypoint_list

        return base_lane

    def keep_speed(self, base_lane, velocity):

        for i in range(len(base_lane.waypoints)):
            set_waypoint_velocity(base_lane.waypoints, i, velocity)

        return base_lane

    def accelerate(self, base_lane, end_velocity, ramp_length):

        return base_lane

    def braking(self, base_lane, stop_index, curr_vel):
        rospy.loginfo("Stopping at waypoint %d. (Current velocity=%f)", stop_index, curr_vel)
        for i in range(stop_index+1):
            wp_distance = distance(base_lane.waypoints, i, stop_index) - self.stopping_dist
            wp_distance = max(0, wp_distance)
            velocity = min(curr_vel, math.sqrt(2 * wp_distance * self.max_decel))
            set_waypoint_velocity(base_lane.waypoints, i, velocity)  

        for i in range(stop_index+1, len(base_lane.waypoints)):
            set_waypoint_velocity(base_lane.waypoints, i, 0)

        return base_lane


    def find_index(self, position, waypoint_list):
        min_index = -1
        min_dist = 9999999.

        for i in range(len(waypoint_list)):
            wp_position = waypoint_list[i].pose.pose.position
            dist = math.sqrt((position.x - wp_position.x)**2 + (position.y - wp_position.y)**2)
            if(dist < min_dist):
                min_dist = dist
                min_index = i

        return [min_dist, min_index]

    def closest_index(self, traffic_light):
        """ Verifies if the traffic light is on the current lookahead path"""

        t_position = traffic_light.pose.pose.position
        wps = self.lane.waypoints
        for i in range(len(wps)):
            wp_position = wps[i].pose.pose.position
            dist = math.sqrt((t_position.x - wp_position.x)**2 + (t_position.y - wp_position.y)**2)
            if(dist < self.dist_threshold):
                return i

        return -1

    def gt_traffic_cb(self, msg):
        """Callback for the ground truth traffic_lights message."""
        self.traffic_lights = msg.lights

    def traffic_cb(self, msg):
        """Callback for the traffic_waypoint message."""
        self.traffic_light_wp_index = msg.data

if __name__ == '__main__':
    try:
        rospy.init_node('waypoint_updater')
        node = WaypointUpdater()
        while not rospy.is_shutdown():
            rospy.spin()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
