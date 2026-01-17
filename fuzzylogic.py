import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

def fuzzify():
    # Inputs: Distance and Angle
    distance = ctrl.Antecedent(np.linspace(0, 10, 100), 'distance')
    angle = ctrl.Antecedent(np.linspace(-180, 180, 100), 'angle')

    # Outputs: Left and Right motor speeds
    left_motor = ctrl.Consequent(np.linspace(0, 100, 100), 'left_motor', 'sugeno')
    right_motor = ctrl.Consequent(np.linspace(0, 100, 100), 'right_motor', 'sugeno')

    # Membership functions for distance
    distance['very_close'] = fuzz.trimf(distance.universe, [0, 0, 1])
    distance['close'] = fuzz.trimf(distance.universe, [0, 1, 2])
    distance['medium_close'] = fuzz.trimf(distance.universe, [1, 2, 3])
    distance['medium'] = fuzz.trimf(distance.universe, [2, 3, 5])
    distance['medium_far'] = fuzz.trimf(distance.universe, [3, 5, 7])
    distance['far'] = fuzz.trimf(distance.universe, [5, 7, 10])
    distance['very_far'] = fuzz.trimf(distance.universe, [7, 10, 10])

    # Membership functions for angle
    angle['sharp_left'] = fuzz.trimf(angle.universe, [-180, -180, -90])
    angle['left'] = fuzz.trimf(angle.universe, [-180, -90, 0])
    angle['slightly_left'] = fuzz.trimf(angle.universe, [-90, 0, 90])
    angle['straight'] = fuzz.trimf(angle.universe, [-30, 0, 30])
    angle['slightly_right'] = fuzz.trimf(angle.universe, [0, 90, 180])
    angle['right'] = fuzz.trimf(angle.universe, [90, 180, 180])
    angle['sharp_right'] = fuzz.trimf(angle.universe, [90, 180, 180])

    # Sugeno-style outputs (7 membership functions for each motor)
    left_motor['left_zero'] = lambda inputs: 0
    left_motor['left_small'] = lambda inputs: 15 + 0.2 * inputs[0] - 0.1 * inputs[1]
    left_motor['left_near_medium'] = lambda inputs: 30 + 0.3 * inputs[0] - 0.1 * inputs[1]
    left_motor['left_medium'] = lambda inputs: 50 + 0.4 * inputs[0] - 0.2 * inputs[1]
    left_motor['left_near_high'] = lambda inputs: 65 - 0.3 * inputs[0] + 0.1 * inputs[1]
    left_motor['left_high'] = lambda inputs: 80 - 0.2 * inputs[0] + 0.1 * inputs[1]
    left_motor['left_very_high'] = lambda inputs: 95 - 0.1 * inputs[0] + 0.2 * inputs[1]

    right_motor['right_zero'] = lambda inputs: 0
    right_motor['right_small'] = lambda inputs: 15 - 0.2 * inputs[0] + 0.1 * inputs[1]
    right_motor['right_near_medium'] = lambda inputs: 30 - 0.3 * inputs[0] + 0.1 * inputs[1]
    right_motor['right_medium'] = lambda inputs: 50 - 0.4 * inputs[0] + 0.2 * inputs[1]
    right_motor['right_near_high'] = lambda inputs: 65 + 0.3 * inputs[0] - 0.1 * inputs[1]
    right_motor['right_high'] = lambda inputs: 80 + 0.2 * inputs[0] - 0.1 * inputs[1]
    right_motor['right_very_high'] = lambda inputs: 95 + 0.1 * inputs[0] - 0.2 * inputs[1]

    return distance, angle, left_motor, right_motor

def create_rules(distance, angle, left_motor, right_motor):
    rules = []
    distance_levels = ['very_close', 'close', 'medium_close', 'medium', 'medium_far', 'far', 'very_far']
    angle_levels = ['sharp_left', 'left', 'slightly_left', 'straight', 'slightly_right', 'right', 'sharp_right']
    left_motor_levels = ['left_zero', 'left_small', 'left_near_medium', 'left_medium', 'left_near_high', 'left_high', 'left_very_high']
    right_motor_levels = ['right_zero', 'right_small', 'right_near_medium', 'right_medium', 'right_near_high', 'right_high', 'right_very_high']

    for i, d in enumerate(distance_levels):
        for j, a in enumerate(angle_levels):
            rules.append(ctrl.Rule(distance[d] & angle[a], (left_motor[left_motor_levels[i]], right_motor[right_motor_levels[j]])))

    return rules

def run_fuzzy_logic():
    rospy.init_node('fuzzy_logic_navigation')
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    distance, angle, left_motor, right_motor = fuzzify()
    rules = create_rules(distance, angle, left_motor, right_motor)
 
    system = ctrl.ControlSystem(rules)
    simulation = ctrl.ControlSystemSimulation(system)

    def scan_callback(scan):
        ranges = np.array(scan.ranges)
        closest_distance = np.min(ranges)
        closest_angle = np.argmin(ranges)

        simulation.input['distance'] = closest_distance
        simulation.input['angle'] = closest_angle

        simulation.compute()

        left_speed = simulation.output['left_motor']
        right_speed = simulation.output['right_motor']

        twist = Twist()
        twist.linear.x = (left_speed + right_speed) / 2
        twist.angular.z = (right_speed - left_speed) / 2
        pub.publish(twist)

    rospy.Subscriber('/scan', LaserScan, scan_callback)
    rospy.spin()
 
if __name__ == '__main__':
    try:
        run_fuzzy_logic()
    except rospy.ROSInterruptException:
        pass
