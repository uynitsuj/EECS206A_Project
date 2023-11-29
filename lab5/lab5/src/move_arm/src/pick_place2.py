#!/usr/bin/env python
import rospy
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest, GetPositionIKResponse
from geometry_msgs.msg import PoseStamped
from moveit_commander import MoveGroupCommander
import numpy as np
from numpy import linalg
import sys
from intera_interface import gripper as robot_gripper


def main():
    # Wait for the IK service to become available
    rospy.wait_for_service('compute_ik')
    rospy.init_node('service_query')
    # Create the function used to call the service
    compute_ik = rospy.ServiceProxy('compute_ik', GetPositionIK)
    # Set up the right gripper
    right_gripper = robot_gripper.Gripper('right_gripper')
    # Calibrate the gripper (other commands won't work unless you do this first)
    print('Calibrating...')
    right_gripper.calibrate()
    rospy.sleep(2.0)
    # Open the right gripper
    print('Opening...')
    right_gripper.open()
    rospy.sleep(1.0)
    while not rospy.is_shutdown():
        input('Press [ Enter ]: ')
        
        # Construct the request
        request = GetPositionIKRequest()
        request.ik_request.group_name = "right_arm"

        # If a Sawyer does not have a gripper, replace '_gripper_tip' with '_wrist' instead
        link = "right_gripper_tip"

        request.ik_request.ik_link_name = link
        # request.ik_request.attempts = 20
        request.ik_request.pose_stamped.header.frame_id = "base"
        
        # Set the desired orientation for the end effector HERE
        request.ik_request.pose_stamped.pose.position.x = 0.884
        request.ik_request.pose_stamped.pose.position.y = 0.067
        request.ik_request.pose_stamped.pose.position.z = -0.159     
        request.ik_request.pose_stamped.pose.orientation.x = -0.015
        request.ik_request.pose_stamped.pose.orientation.y = 0.998
        request.ik_request.pose_stamped.pose.orientation.z = -0.025
        request.ik_request.pose_stamped.pose.orientation.w = 0.062
        
        try:
            # Send the request to the service
            response = compute_ik(request)
            
            # Print the response HERE
            print(response)
            group = MoveGroupCommander("right_arm")

            # Setting position and orientation target
            group.set_pose_target(request.ik_request.pose_stamped)

            # TRY THIS
            # Setting just the position without specifying the orientation
            ###group.set_position_target([0.5, 0.5, 0.0])

            # Plan IK
            plan = group.plan()
            user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")
            
            # Execute IK if safe
            if user_input == 'y':
                group.execute(plan[1])
            
            
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        

        # Close the right gripper
        print('Closing...')
        right_gripper.close()
        rospy.sleep(1.0)
        

        ''' # Intermediary information
        request.ik_request.pose_stamped.pose.position.x = 0.855
        request.ik_request.pose_stamped.pose.position.y = -0.335
        request.ik_request.pose_stamped.pose.position.z = -0.053   
        request.ik_request.pose_stamped.pose.orientation.x = 0.062
        request.ik_request.pose_stamped.pose.orientation.y = 0.995
        request.ik_request.pose_stamped.pose.orientation.z = 0.021
        request.ik_request.pose_stamped.pose.orientation.w = 0.076

        try:
            # Send the request to the service
            response = compute_ik(request)
            
            # Print the response HERE
            print(response)
            group = MoveGroupCommander("right_arm")

            # Setting position and orientation target
            group.set_pose_target(request.ik_request.pose_stamped)

            # TRY THIS
            # Setting just the position without specifying the orientation
            ###group.set_position_target([0.5, 0.5, 0.0])

            # Plan IK
            plan = group.plan()
            user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")
            
            # Execute IK if safe
            if user_input == 'y':
                group.execute(plan[1])
            
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)'''
        

        # Place information
        request.ik_request.pose_stamped.pose.position.x = 0.800
        request.ik_request.pose_stamped.pose.position.y = -0.355
        # request.ik_request.pose_stamped.pose.position.z = -0.165
        request.ik_request.pose_stamped.pose.position.z = -0.1   
        request.ik_request.pose_stamped.pose.orientation.x = 0.035
        request.ik_request.pose_stamped.pose.orientation.y = 0.998
        request.ik_request.pose_stamped.pose.orientation.z = -0.007
        request.ik_request.pose_stamped.pose.orientation.w = 0.054

        try:
            # Send the request to the service
            response = compute_ik(request)
            
            # Print the response HERE
            print(response)
            group = MoveGroupCommander("right_arm")

            # Setting position and orientation target
            group.set_pose_target(request.ik_request.pose_stamped)

            # TRY THIS
            # Setting just the position without specifying the orientation
            ###group.set_position_target([0.5, 0.5, 0.0])

            # Plan IK
            plan = group.plan()
            user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")
            
            # Execute IK if safe
            if user_input == 'y':
                group.execute(plan[1])
            
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        

        # Open the right gripper
        print('Opening...')
        right_gripper.open()
        rospy.sleep(1.0)
        print('Done!')
        

        

# Python's syntax for a main() method
if __name__ == '__main__':
    main()
