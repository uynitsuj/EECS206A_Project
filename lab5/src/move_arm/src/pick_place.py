#!/usr/bin/env python
import rospy
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest, GetPositionIKResponse
from geometry_msgs.msg import PoseStamped
from moveit_commander import MoveGroupCommander
import numpy as np
from numpy import linalg
import sys
from intera_interface import gripper as robot_gripper


def pick_ik_request(compute_ik):
    # Construct the pick_request
    pick_request = GetPositionIKRequest()
    pick_request.ik_request.group_name = "right_arm"

    # If a Sawyer does not have a gripper, replace '_gripper_tip' with '_wrist' instead
    link = "right_gripper_tip"

    pick_request.ik_request.ik_link_name = link
    # pick_request.ik_request.attempts = 20
    pick_request.ik_request.pose_stamped.header.frame_id = "base"
    
    # Pick information
    pick_request.ik_request.pose_stamped.pose.position.x = 0.884
    pick_request.ik_request.pose_stamped.pose.position.y = 0.067
    pick_request.ik_request.pose_stamped.pose.position.z = -0.159     
    pick_request.ik_request.pose_stamped.pose.orientation.x = -0.015
    pick_request.ik_request.pose_stamped.pose.orientation.y = 0.998
    pick_request.ik_request.pose_stamped.pose.orientation.z = -0.025
    pick_request.ik_request.pose_stamped.pose.orientation.w = 0.062
    
    try:
        # Send the pick_request to the service
        response = compute_ik(pick_request)
        
        # Print the response HERE
        print(response)
        group = MoveGroupCommander("right_arm")

        # Setting position and orientation target
        group.set_pose_target(pick_request.ik_request.pose_stamped)

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

def inter_ik_request(compute_ik):
    # Construct the place_request
    place_request = GetPositionIKRequest()
    place_request.ik_request.group_name = "right_arm"

    # If a Sawyer does not have a gripper, replace '_gripper_tip' with '_wrist' instead
    link = "right_gripper_tip"

    place_request.ik_request.ik_link_name = link
    # place_request.ik_request.attempts = 20
    place_request.ik_request.pose_stamped.header.frame_id = "base"
    
    # Place information
    place_request.ik_request.pose_stamped.pose.position.x = 0.855
    place_request.ik_request.pose_stamped.pose.position.y = -0.335
    place_request.ik_request.pose_stamped.pose.position.z = -0.053   
    place_request.ik_request.pose_stamped.pose.orientation.x = 0.062
    place_request.ik_request.pose_stamped.pose.orientation.y = 0.995
    place_request.ik_request.pose_stamped.pose.orientation.z = 0.021
    place_request.ik_request.pose_stamped.pose.orientation.w = 0.076
    
    try:
        # Send the place_request to the service
        response = compute_ik(place_request)
        
        # Print the response HERE
        print(response)
        group = MoveGroupCommander("right_arm")

        # Setting position and orientation target
        group.set_pose_target(place_request.ik_request.pose_stamped)

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

def place_ik_request(compute_ik):
    # Construct the place_request
    place_request = GetPositionIKRequest()
    place_request.ik_request.group_name = "right_arm"

    # If a Sawyer does not have a gripper, replace '_gripper_tip' with '_wrist' instead
    link = "right_gripper_tip"

    place_request.ik_request.ik_link_name = link
    # place_request.ik_request.attempts = 20
    place_request.ik_request.pose_stamped.header.frame_id = "base"
    
    # Place information
    place_request.ik_request.pose_stamped.pose.position.x = 0.800
    place_request.ik_request.pose_stamped.pose.position.y = -0.355
    place_request.ik_request.pose_stamped.pose.position.z = -0.165   
    place_request.ik_request.pose_stamped.pose.orientation.x = 0.035
    place_request.ik_request.pose_stamped.pose.orientation.y = 0.998
    place_request.ik_request.pose_stamped.pose.orientation.z = -0.007
    place_request.ik_request.pose_stamped.pose.orientation.w = 0.054
    
    try:
        # Send the place_request to the service
        response = compute_ik(place_request)
        
        # Print the response HERE
        print(response)
        group = MoveGroupCommander("right_arm")

        # Setting position and orientation target
        group.set_pose_target(place_request.ik_request.pose_stamped)

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


def close_gripper(right_gripper):
    # Close the right gripper
    print('Closing...')
    right_gripper.close()
    rospy.sleep(1.0)

def open_gripper(right_gripper):
    # Open the right gripper
    print('Opening...')
    right_gripper.open()
    rospy.sleep(1.0)
    print('Done!')


def main():
    # Wait for the IK service to become available
    rospy.wait_for_service('compute_ik')
    rospy.init_node('service_query')
    # Create the function used to call the service
    compute_ik = rospy.ServiceProxy('compute_ik', GetPositionIK)
    # Set up the right gripper
    right_gripper = robot_gripper.Gripper('right_gripper')
    while not rospy.is_shutdown():
        input('Press [ Enter ]: ')
        # open_gripper(right_gripper)
        # input('Press [ Enter ]: ')
        pick_ik_request(compute_ik)
        # input('Press [ Enter ]: ')
        close_gripper(right_gripper)
        # input('Press [ Enter ]: ')
        inter_ik_request(compute_ik)
        place_ik_request(compute_ik)
        # input('Press [ Enter ]: ')
        open_gripper(right_gripper)
        
        

# Python's syntax for a main() method
if __name__ == '__main__':
    main()
