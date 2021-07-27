import tf2_ros
import tf2_geometry_msgs
from tf2_geometry_msgs import PoseStamped
from tf.transformations import euler_from_quaternion
import math
import pandas as pd
import numpy as np
import csv
import sys
sys.path.insert(0, "./trajectron")
from environment import Environment, Scene, derivative_of, Node


def create_scene(buffer, scene_ids, present_id):
    obstacle_id_list = []
    buffer_data = buffer.buffer_frame
    max_timesteps = buffer_data['frame_id'].max()
    scene = Scene(timesteps=max_timesteps + 1, dt=0.5)
    for node_id in scene_ids:
        node_frequency_multiplier = 1
        node_df = buffer_data[buffer_data['node_id'] == node_id]

        if node_df['x'].shape[0] < 2:
            continue

        if not np.all(np.diff(node_df['frame_id']) == 1):
            # print('Occlusion')
            # print 'here!'
            continue  # TODO Make better

        node_values = node_df[['x', 'y']].values
        x = node_values[:, 0]
        y = node_values[:, 1]
        heading = node_df['heading_ang'].values
        # TODO get obstacle id
        vx = derivative_of(x, scene.dt)
        vy = derivative_of(y, scene.dt)

        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)
        if node_df.iloc[0]['type'] == buffer.env.NodeType.VEHICLE:
            v = np.stack((vx, vy), axis=-1)
            v_norm = np.linalg.norm(
                np.stack((vx, vy), axis=-1), axis=-1, keepdims=True)

            if np.sum(v_norm, axis=0)[
                    0] / len(v_norm) < 0.2 and node_id in present_id:
                # print('v_norm : ',np.sum(v_norm, axis = 0)[0]/len(v_norm))
                obstacle_id_list.append(int(node_id))

            heading_v = np.divide(
                v, v_norm, out=np.zeros_like(v), where=(
                    v_norm > 0.1))
            heading_x = heading_v[:, 0]
            heading_y = heading_v[:, 1]

            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('velocity', 'norm'): np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1),
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay,
                         ('acceleration', 'norm'): np.linalg.norm(np.stack((ax, ay), axis=-1), axis=-1),
                         ('heading', 'x'): heading_x,
                         ('heading', 'y'): heading_y,
                         ('heading', 'angle'): heading,
                         ('heading', 'radian'): node_df['heading_rad'].values}
            node_data = pd.DataFrame(
                data_dict, columns=buffer.data_columns_vehicle)
            output_node_data = node_data
            output_node_data = node_data['frame_id'] = buffer.get_curr_frame()
            # print('node_data : ',output_node_data)
        else:
            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay}
            node_data = pd.DataFrame(
                data_dict, columns=buffer.data_columns_pedestrian)

        node = Node(
            node_type=node_df.iloc[0]['type'],
            node_id=node_id,
            data=node_data,
            frequency_multiplier=node_frequency_multiplier)
        node.first_timestep = node_df['frame_id'].iloc[0]
        scene.nodes.append(node)
    buffer.update_prev_id(obstacle_id_list)
    return scene


def transform_data(buffer, data, tf_map, tf_buffer, rospy):
    present_id_list = []
    for obj in data.objects:
        # category = None
        # if obj.classId == 1: #temporary test
        #     category = buffer.env.NodeType.VEHICLE
        #     type_ = "PEDESTRIAN"
        # elif obj.classId == 2 or obj.classId == 3 or obj.classId == 4:
        #     category = buffer.env.NodeType.VEHICLE
        #     type_ = "VEHICLE"
        # else:
        #     continue
        id = int(obj.track.id)
        category = buffer.env.NodeType.VEHICLE
        type_ = "VEHICLE"
        x = obj.center_point.x
        y = obj.center_point.y
        z = obj.center_point.z
        # transform from base_link to map
        if tf_map:
            transform = tf_buffer.lookup_transform(
                'map', 'base_link', rospy.Time(0), rospy.Duration(1.0))
            pose_stamped = PoseStamped()
            pose_stamped.pose.position.x = x
            pose_stamped.pose.position.y = y
            pose_stamped.pose.position.z = z
            pose_transformed = tf2_geometry_msgs.do_transform_pose(
                pose_stamped, transform)
            x = pose_transformed.pose.position.x
            y = pose_transformed.pose.position.y
            z = pose_transformed.pose.position.z
        # print 'x: ', x
        # print(pose_transformed.pose.position.x, pose_transformed.pose.position.y, pose_transformed.pose.position.z)

        length = math.sqrt(
            math.pow(
                (obj.bPoint.p4.x -
                 obj.bPoint.p0.x),
                2) +
            math.pow(
                (obj.bPoint.p4.y -
                 obj.bPoint.p0.y),
                2))
        width = math.sqrt(
            math.pow(
                (obj.bPoint.p3.x -
                 obj.bPoint.p0.x),
                2) +
            math.pow(
                (obj.bPoint.p3.y -
                 obj.bPoint.p0.y),
                2))
        height = math.sqrt(
            math.pow(
                (obj.bPoint.p1.x -
                 obj.bPoint.p0.x),
                2) +
            math.pow(
                (obj.bPoint.p1.y -
                 obj.bPoint.p0.y),
                2))
        diff_x = 0
        diff_y = 0
        heading = 0
        # for CH object.yaw
        heading = obj.distance
        heading_rad = math.radians(heading)
        # heading_rad = 0
        # for old_obj in past_obj:
        #     sp = old_obj.split(",")
        #     if obj.track.id == int(sp[2]):
        #         diff_x = x - float(sp[4])
        #         diff_y = y - float(sp[5])
        #         if diff_x == 0:
        #             heading = 90
        #         else:
        #             heading = abs(math.degrees(math.atan(diff_y / diff_x)))
        #         # print(diff_x,diff_y,diff_y/diff_x,heading)
        #         if diff_x == 0 and diff_y == 0:
        #             heading = 0
        #         elif diff_x >= 0 and diff_y >= 0:
        #             heading = heading
        #         elif diff_x >= 0 and diff_y < 0:
        #             heading = 360 - heading
        #         elif diff_x < 0 and diff_y >= 0:
        #             heading = 180 - heading
        #         else:
        #             heading = 180 + heading
        #         if heading > 180:
        #             heading = heading - 360
        #         heading_rad = math.radians(heading)
        # info = str(buffer.get_buffer_frame()) + "," + type_ + "," + str(obj.track.id) + "," + "False" + "," + str(x) + \
        #     "," + str(y) + "," + str(z) + "," + str(length) + "," + str(width) + "," + str(height) + "," + str(heading)
        # past_obj.append(info)
        # print 'ros method heading : ',yaw
        # print 'our method heading : ',heading
        node_data = pd.Series({'frame_id': buffer.get_curr_frame(),
                               'type': category,
                               'node_id': str(obj.track.id),
                               'robot': False,  # frame_data.loc[i]['robot']
                               'x': x,
                               'y': y,
                               'z': z,
                               'length': length,
                               'width': width,
                               'height': height,
                               'heading_ang': heading,
                               'heading_rad': heading_rad})
        buffer.update_buffer(node_data)
        present_id_list.append(id)
        # print(id)

    # add temp data
    mask_id_list = buffer.add_temp_obstacle(present_id_list)
    # print(present_id_list)
    buffer.refresh_buffer()
    # buffer.add_frame_length(len(present_id_list))
    buffer.add_frame_length(len(present_id_list) + len(mask_id_list))
    return present_id_list, mask_id_list


def output_csvfile(data_path, csv_filename, data):
    data.to_csv(data_path + csv_filename)
