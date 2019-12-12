#!/usr/bin/env python
import rospy
import yaml
import numpy as np
import cv2
from matplotlib import pyplot as plt
#
from nav_msgs.msg import (
    OccupancyGrid,
)
from map_msgs.msg import (
    OccupancyGridUpdate,
)


# Extra utility functions
# import utility as utl


# import tf
from tf import transformations
import tf2_ros


# Utilities
#----------------------------------------------------------#
def transform_2_translation_quaternion(transform_m):
    """
    This function help transfer the geometry_msgs.msg.TransformStamped
    into (translation, quaternion) <-- lists
    """
    # trans_v = transform_m.transform.translation
    # quat_v = transform_m.transform.rotation
    #
    trans = [transform_m.transform.translation.x, transform_m.transform.translation.y, transform_m.transform.translation.z]
    quaternion = [transform_m.transform.rotation.x, transform_m.transform.rotation.y, transform_m.transform.rotation.z, transform_m.transform.rotation.w]
    # trans = np.array([transform_m.transform.translation.x, transform_m.transform.translation.y, transform_m.transform.translation.z])
    # quaternion = np.array([transform_m.transform.rotation.x, transform_m.transform.rotation.y, transform_m.transform.rotation.z, transform_m.transform.rotation.w])
    return (trans, quaternion)
#----------------------------------------------------------#

class COSTMAP_LISTENER(object):
    """
    This is the class for costmap_listener, where the costmap is generated by another node and published by topics.
    """
    def __init__(self, param_dict, tf_buffer=None):
        """
        inputs
            - param_dict
            - tf_buffer: tf2_ros buffer object

        The param_dict should include the following fields:
        - costmap_ns: {"local_costmap", "global_costmap"}
        - map_frame_id
        - base_frame_id
        - is_using_external_footprint (default: False)
        - footprint (optional): a list of (x,y) pairs, only used if "is_using_external_footprint" is True
        """
        # self.costmap_ns = costmap_ns
        # costmap_ns = "/move_base/local_costmap"
        # costmap_ns = "/move_base/global_costmap"

        self.tf_buffer = tf_buffer

        # Get parameters from param_dict
        self.costmap_ns = param_dict.get('costmap_ns', "/move_base/local_costmap") # or, "/move_base/global_costmap"
        self.map_frame = param_dict.get('map_frame_id', "map") # "map"
        self.base_frame = param_dict.get('base_frame_id', "base_footprint") # "base_footprint"
        is_using_external_footprint = param_dict.get('is_using_external_footprint', False)


        # Getting footprint, vertexes of the polygon
        #----------------------------------------------------#
        _footprint = [[-0.57, 0.36],[0.57, 0.36],[0.57, -0.36],[-0.57, -0.36]]
        if is_using_external_footprint:
            print('[Costmap-listener] Use external footprint.')
            _footprint = param_dict.get('footprint', _footprint ) # A list of point2D
        else:
            # paramters load from rosparam server
            is_parameters_set = False
            _count_tried = 0
            while (not is_parameters_set) and (_count_tried < 100) and (not rospy.is_shutdown()):
                _count_tried += 1
                try:
                    _footprint = yaml.load( rospy.get_param(self.costmap_ns + "/footprint") )
                    is_parameters_set = True
                except:
                    rospy.loginfo("footprint parameters are not found in rosparam server, keep on trying...")
                    rospy.sleep(0.2) # Sleep 0.2 seconds for waiting the parameters loading
                    continue
            print('[Costmap-listener] "footprint" was %sfound in parameter server.' % ('' if is_parameters_set else 'not '))
        #
        print( "[Costmap-listener] footprint %s = %s" % ( type(_footprint), str(_footprint) ) )

        # Make this a list of numpy array ( shape=(2,) ), which is called point2D
        self.footprint = list()
        try:
            for point in _footprint:
                self.footprint.append(np.array(point, dtype=float).reshape((2,)))
        except:
            # Somthing wrong in making footprint to np-array
            rospy.logerr("[Costmap-listener] Somthing wrong in making footprint points into numpy-array")
            pass
        # print("[Costmap-listener] self.footprint = " + str(self.footprint))
        #----------------------------------------------------#
        # end Getting footprint

        # Parameters
        # Special values
        self.special_values_lethal = 100 # Maximum value, which means an obstacle is on that point
        self.special_values_inscribed = 99
        self.special_values_free = 0
        self.special_values_unknown = -1

        # States
        self.costmap_initiated = False # **[This flag will be updated only once.]** The original costmap should be loaded first before we can start updating
        self.costmap_updated = False
        #
        self.update_stamp = None
        self.costmap_frame = None
        self.costmap_info = None
        """
        info:
            - map_load_time
            - resolution
            - width
            - height
            - origin
        """
        self.costmap_shape = np.array((0,0))
        self.origin_point2D = np.array((0,0)) # shape = (2,). There's no need to get the orientation, since it's always be 0 deg.
        # This is the data of costmap, shape=(height, width)
        self.costmap = np.zeros((1,1))



        # Robot
        self.robot_pose2D = np.zeros((3,1))

        # ROS subscriber
        rospy.Subscriber("%s/costmap" % self.costmap_ns, OccupancyGrid, self._costmap_CB)
        rospy.Subscriber("%s/costmap_updates" % self.costmap_ns, OccupancyGridUpdate, self._costmap_update_CB)



    def _costmap_CB(self, data):
        """
        This is the callback function for costmap.
        """
        self.update_stamp = data.header.stamp
        self.costmap_frame = data.header.frame_id
        self.costmap_info = data.info
        self.costmap_shape = np.array( (data.info.height, data.info.width) ) # (height,width); height for y-direction, width for x-direction, like image
        self.origin_point2D = np.array( (data.info.origin.position.x, data.info.origin.position.y) ) # shape = (2,). There's no need to get the orientation, since it's always be 0 deg.
        # print self.costmap_info
        #
        # costmap_array = np.array(data.data, dtype=int)
        self.costmap = np.array(data.data, dtype=int).reshape(self.costmap_shape)
        # Note: the (0,0) is the point closest to the origin.
        if not self.costmap_updated:
            self.costmap_updated = True
        #
        if not self.costmap_initiated:
            self.costmap_initiated = True



    def _costmap_update_CB(self, data):
        """
        This is the call back function for costmap_update.
        """
        # We cannot update until we got the full costmap
        if not self.costmap_initiated:
            return
        #
        self.update_stamp = data.header.stamp
        self.costmap_frame = data.header.frame_id
        #
        pose = (data.y, data.x) # row is y, column is x
        width = data.width # x-direction
        height = data.height # y-direction
        # costmap_array = np.array(data.data, dtype=int)
        # costmap_tmp = costmap_array.reshape((height,width))
        # Note: the (0,0) is the point closest to the origin.
        corners = (pose[0], pose[0]+height, pose[1], pose[1]+width)
        self.costmap[corners[0]:corners[1], corners[2]:corners[3]] = np.array(data.data, dtype=int).reshape((height,width)) # costmap_tmp
        # print("pose = %s" % str(pose))
        # print("(height, width) = (%d, %d)" % (height, width))

        if not self.costmap_updated:
            self.costmap_updated = True

    def _get_point2D_from_index2D(self, index2D):
        """
        This function return the point2D from the given index2D.
        inputs
            - index2D: (m in row, n in col) <-- (y,x), 0-indexed
        outputs
            - point2D: (x,y), shape=(2,), 1-D array,
                      ** Note: it's not pose2D, which is a 3x1 2-D array.
        """
        # We cannot calculate until we got the full costmap
        if not self.costmap_initiated:
            return None

        index2D_np = np.array(index2D).reshape((2,))
        # Get the point2D: (x, y) from (m,n), 0-indexed
        # First flip the order in index
        # Then shift to the center of a cell by adding 0.5
        # Change to the correct unit by multiplying resolution
        # Shift to the refeerence frame by adding the origin of the costmap
        return ( self.origin_point2D + ( index2D_np[::-1,].astype(float) + 0.5 )*self.costmap_info.resolution )

    def _get_index2D_from_point2D(self, point2D, is_checking_bound=True):
        """
        This function return the index2D from the given pose2D.
        inputs
            - point2D: (x,y), shape=(2,), 1-D array,
                      ** Note: it's not pose2D, which is a 3x1 2-D array.
        outputs
            - index2D: (m in row, n in col) <-- (y,x), 0-indexed
                      ** Note: The returned index2D might be out of the bound of the costmap!!
                               The reason for returning an invalid index is for calculating a line toward the bound.
        """
        # We cannot calculate until we got the full costmap
        if not self.costmap_initiated:
            return None

        try:
            resolution_inv = 1.0/(self.costmap_info.resolution) # cell/m
        except:
            # self.costmap_info.resolution is zero
            rospy.logerr("[Costmap-listener] Resolution is 0.")
            return None
        point2D_np = np.array(point2D).reshape((2,))
        # Get the index2D: (m, n) from (x,y), 0-indexed
        # First calculate the relative position
        # Then digitalized, including unit-change, floor-rounding, and converting to "int"
        # The last commad help flip the array in first dimension, so that we get the order in (m,n)
        index2D_np = np.floor((point2D_np - self.origin_point2D)*resolution_inv).astype(int)[::-1,]

        if is_checking_bound:
            if (np.sum(index2D_np < 0) > 0) or (np.sum(index2D_np >= self.costmap_shape) > 0):
                # out of bound
                rospy.logwarn("[Costmap-listener] The point queried is out of bound.")
                return None
                # Simply warning, still return an invalid index
            #
        #
        return index2D_np


    # Transformations
    #-------------------------------------------#
    def _transform2D_point2D(self, tf2D_sb, point2D_b):
        """
        This method transform a point2D in frame {b} in to frame {s},
        according to the pose2D of frame {b} relative to {s} <-- tf2D_sb

        ** Note: pose2D_sb (pose2D of {b} represented by {b}) is equivalent to tf2D_sb (transform of {b} relative to {s})

        inputs
            - tf2D_sb: shape=(3,1), tf2D_sb := [[x,y,theta]]
            - point2D_b: shape=(2,), point2D_b := [x,y]
        outputs
            - point2D_s: shape=(2,), point2D_s := [x,y]
        """
        point2D_s = np.zeros((2,))
        ct = np.cos(tf2D_sb[2,0])
        st = np.sin(tf2D_sb[2,0])
        point2D_s[0] = ct*point2D_b[0] - st*point2D_b[1] + tf2D_sb[0,0]
        point2D_s[1] = st*point2D_b[0] + ct*point2D_b[1] + tf2D_sb[1,0]
        #
        return point2D_s

    def _get_transformed_footprint(self, pose2D):
        """
        This method transform the footprint (a list of point2Ds, represented in base-frame) from base-frame to reference frame
        by given pose2D (shape=(3,1)).

        inputs
            - pose2D: The pose2D of the base-frame relative to the reference frame
        outputs
            - footprint_ref: The footprint represented in the reference frame
        """
        footprint_base = self.footprint
        footprint_ref = list()
        for point2D_base in footprint_base:
            # Transform2D: {base} --> {ref.}
            point2D_ref = self._transform2D_point2D(pose2D, point2D_base)
            footprint_ref.append(point2D_ref)
        return footprint_ref
    #-------------------------------------------#


    # Methods for calculating the cost
    #-------------------------------------------#
    def _get_line_of_index2Ds(self, index2D_pair):
        """
        This is the function for getting all the points in a line segment.
        Say the given end-point pair is (a,b),
        the output line is a region of [a,b), where the last point is not included.

        inputs
            - index2D_pair: a pair (tuple) of index2Ds, which are indexes of end points.
        outputs
            - index2D_list: a list of index2D which form a line, and points are 8-neighbor connected to each other.
                        ** Note: the last point is not included,
                                 which means that this line is a regin of [a,b)
        """
        try:
            index2D_a = np.array(index2D_pair[0], dtype=int)
            index2D_b = np.array(index2D_pair[1], dtype=int)
        except:
            # Might be at least one of them is None or something else, whatever
            if len(index2D_pair) < 2:
                return []
            #
            if index2D_pair[0] is None:
                return []
            else:
                return [index2D_a]
        #
        vector_ab = (index2D_b - index2D_a) # still with dtype int
        idx_primary_axis = np.argmax(np.absolute(vector_ab)) # Find the primary axis
        #
        primary_length = np.absolute(vector_ab[idx_primary_axis]) # Positive value
        if primary_length == 0:
            # Simply a point
            return [index2D_a]
        #
        direction_ab = vector_ab.astype(float)/primary_length # Note: the element at idx_primary_axis is now with length 1.0 (1.0 or -1.0)
        index2D_list = list()
        # Iterate through the primary axis
        for mu in range(primary_length):
            # Note: the last point is not included
            index2D_list.append(index2D_a + np.around(mu*direction_ab).astype(int)) # Rounding and change type
        #
        return index2D_list

    def _get_contour_of_polygon(self, index2D_vertexes):
        """
        Given a list of index2D (vertexes), return the contour, which is a list of index2D.
        Say the given vertexs are (a,b,c),
        the output contour is form by segments [a,b), [b,c), [c,a).

        inputs
            - index2D_vertexes: a list of index2Ds, which are(is) vertexes(vertex)
        outputs
            - index2D_list: a list of index2D which form a contour, and points are 8-neighbor connected to each other.
        """
        num_vertexes = len(index2D_vertexes) # Also equals to number of line-segments
        if num_vertexes == 0:
            return []
        #
        _index2D_vertexes = index2D_vertexes
        _index2D_vertexes.append(index2D_vertexes[0]) # Add the first point to the end to form a cycle

        # Iterate through line-segments
        index2D_list = list()
        for i in range(num_vertexes):
            # concatenation, not appending
            index2D_list += self._get_line_of_index2Ds( (_index2D_vertexes[i],_index2D_vertexes[i+1]) )
        return index2D_list


    def get_footprint_cost(self, pose2D):
        """'
        This method return the maximum cost of points laid on the contour of footprint,
        where the pose of footprint is given by pose2D of the base-frame in the reference frame.

        inputs
            - pose2D: The pose2D of the base-frame relative to the reference frame
        outputs
            - cost_max: The highest value of points on the contour of footprint.
                    ** Note: If somthing wrong, this value might be None
        """
        # We cannot calculate until we got the full costmap
        if not self.costmap_initiated:
            return None

        # point2D
        footprint_ref = self._get_transformed_footprint(pose2D)
        index2D_vertexes = list()
        for point2D in footprint_ref:
            # Note: not checking bounds
            index2D_vertexes.append(self._get_index2D_from_point2D(point2D, is_checking_bound=False))
        # index2D
        contour_idx = self._get_contour_of_polygon(index2D_vertexes)
        #
        cost_max = -1
        exist_out_of_bound_point = False
        for index2D in contour_idx:
            #
            try:
                if self.costmap[index2D[0], index2D[1]] > cost_max:
                    cost_max = self.costmap[index2D[0], index2D[1]]
                    if cost_max >= self.special_values_lethal:
                        # Reach the maximum value, no need to continue.
                        break
            except:
                # out of bound of costmap
                if not exist_out_of_bound_point:
                    exist_out_of_bound_point = True
                    rospy.logwarn("[Costmap-listener] The footprint is out of the bound of costmap.")
        #
        return cost_max

    def get_current_footprint_cost(self):
        """
        This method return the current footprint cost

        outputs
            - cost_max: The highest value of points on the contour of footprint.
        """
        res = self.get_pose2D_base_in_costmap()
        if res is None:
            return None
        # else
        return self.get_footprint_cost(res[0])
    #-------------------------------------------#

    def get_pose2D_base_in_costmap(self):
        """
        This is the fuction for getting the robot pose (/base_foorprint) referring to /map
        The function is mainly for global-location initialization.
        """
        if self.tf_buffer is None:
            rospy.logwarn("[Costmap-listener] tf_buffer was not provided while trying to get robot pose.")
            return None

        # Get pose_2D from tf to reduce the effect of delay
        # [x, y, theta].'
        pose_2D = np.zeros((3,1))
        #
        # Try tf
        try:
            # From /map to /base_footprint
            # For outputing the stamp
            stamp_amclPose = rospy.Time.now()
            # print "~~~ Delay of the amcl_pose: ", (rospy.Time.now() - stamp_amclPose).to_sec(), "sec."
            transform_m = self.tf_buffer.lookup_transform(self.costmap_frame, self.base_frame, rospy.Time(0))
            # Translation of data types
            (trans, quaternion) = transform_2_translation_quaternion(transform_m)
            #
            pose_2D[0,0] = trans[0] # self._amcl_pose.position.x
            pose_2D[1,0] = trans[1] # self._amcl_pose.position.y
            # pose = self._amcl_pose
            # quaternion = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
            euler = transformations.euler_from_quaternion(quaternion)
        except: # (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            # print "tf exception while getting robot pose"
            rospy.logwarn("[Costmap-listener] tf exception while getting robot pose.")
            #
            return None
        #
        # roll = euler[0]
        # pitch = euler[1]
        # yaw = euler[2]
        pose_2D[2,0] = euler[2] # yaw
        # Reduced-order covariance matrix
        # cov_2D = np.dot(np.dot(self._T_subState, self._amcl_cov), np.transpose(self._T_subState) )
        #
        return (pose_2D, stamp_amclPose)

    def plot_costmap(self, show_obstacle_onlt=False):
        """
        This function help plot the costmap with pyplot
        """
        # Flip the image for maching the image coordinate convention!

        if show_obstacle_onlt:
            Cmap_H = cv2.flip(self.costmap,0)>=100
        else:
            Cmap_H = cv2.flip(self.costmap,0)
        plt.imshow(Cmap_H, cmap = 'gray', interpolation = 'bicubic')
        plt.show(block = False)
        # plt.draw()
        plt.pause(0.001)


def costmap_listener():
    # Init. node
    rospy.init_node('costmap_listener', anonymous=True)

    costmap_ns = "/move_base/local_costmap"
    # costmap_ns = "/move_base/global_costmap"

    # tf listener
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    param_dict = dict()
    param_dict['costmap_ns'] = costmap_ns
    param_dict['map_frame_id'] = 'map'
    param_dict['base_frame_id'] = 'base_footprint'
    param_dict['is_using_external_footprint'] = False
    # param_dict['footprint'] = [[-0.57, 0.36],[0.57, 0.36],[0.57, -0.36],[-0.57, -0.36]] # [(1,1),(1,-1),(-1,-1),(-1,1)]
    #
    costmap_L = COSTMAP_LISTENER(param_dict, tf_buffer)


    rate = rospy.Rate(20) # 20hz
    while not rospy.is_shutdown():
        #
        if costmap_L.costmap_updated:
            # print("costmap_updated")
            costmap_L.plot_costmap(show_obstacle_onlt=False)
            costmap_L.costmap_updated = False
            #
            cost = costmap_L.get_current_footprint_cost()
            print("cost = " + str(cost))
        #
        rate.sleep()


if __name__ == '__main__':
    costmap_listener()
