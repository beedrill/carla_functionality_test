#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Mayavi Lidar visualization example for CARLA"""

import glob
import os
import sys
import argparse
import time
from datetime import datetime
import random
import threading
import numpy as np
from matplotlib import cm
from mayavi import mlab

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

def lidar_callback(point_cloud, buf):
    """Prepares a point cloud with intensity colors and stores in a buffer, which updated the mayavi visualisation"""
    data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))

    # Isolate the intensity and compute a color for it
    intensity = data[:, -1]
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))

    # Isolate the 3D data
    points = data[:, :-1]

    # We're negating the y to correclty visualize a world that matches
    # what we see in Unreal since Mayavi uses a right-handed coordinate system
    # points[:, :1] = -points[:, :1]

    # # An example of converting points from sensor to vehicle space if we had
    # # a carla.Transform variable named "tran":
    # points = np.append(points, np.ones((points.shape[0], 1)), axis=1)
    # points = np.dot(tran.get_matrix(), points.T).T
    # points = points[:, :-1]

    #copy points/intensities into buffer
    buf['pts'] = points
    buf['intensity'] = intensity

def semantic_lidar_callback(point_cloud, buf):
    """Prepares a point cloud with semantic segmentation colors"""
    data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))

    # We're negating the y to correclty visualize a world that matches
    # what we see in Unreal since Open3D uses a right-handed coordinate system
    points = np.array([data['x'], -data['y'], data['z']]).T

    # # An example of adding some noise to our data if needed:
    # points += np.random.uniform(-0.05, 0.05, size=points.shape)

    # Colorize the pointcloud based on the CityScapes color palette
    labels = np.array(data['ObjTag'], dtype=np.float32)
    int_color = labels

    # # In case you want to make the color intensity depending
    # # of the incident ray angle, you can use:
    #  int_color *= np.array(data['CosAngle'])

    buf['pts'] = points
    buf['intensity'] = labels

def generate_lidar_bp(arg, world, blueprint_library, delta):
    """Generates a CARLA blueprint based on the script parameters"""
    if arg.semantic:
        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
    else:
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        if arg.no_noise:
            lidar_bp.set_attribute('dropoff_general_rate', '0.0')
            lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
            lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
        else:
            lidar_bp.set_attribute('noise_stddev', '0.2')

    lidar_bp.set_attribute('upper_fov', str(arg.upper_fov))
    lidar_bp.set_attribute('lower_fov', str(arg.lower_fov))
    lidar_bp.set_attribute('channels', str(arg.channels))
    lidar_bp.set_attribute('range', str(arg.range))
    lidar_bp.set_attribute('rotation_frequency', str(1.0 / delta))
    lidar_bp.set_attribute('points_per_second', str(arg.points_per_second))
    return lidar_bp

def carlaEventLoop(world):
        frame = 0
        dt0 = datetime.now()
        while True:
            time.sleep(0.005)
            world.tick()

            process_time = datetime.now() - dt0
            sys.stdout.write('\r' + 'FPS: ' + str(1.0 / process_time.total_seconds()))
            sys.stdout.flush()
            dt0 = datetime.now()
            frame += 1

def get_bounding_box(vehicle, lidar):
    """
    Returns 3D bounding box for a vehicle based on camera view.
    """

    bb_cords = _create_bb_points(vehicle)
    cords_x_y_z = _vehicle_to_sensor(bb_cords, vehicle, lidar)[:3, :]
    return coords_x_y_z

def _create_bb_points(vehicle):
    """
    Returns 3D bounding box for a vehicle.
    """

    cords = np.zeros((8, 4))
    extent = vehicle.bounding_box.extent
    cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
    cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
    cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
    cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
    cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
    cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
    cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
    cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
    return cords

def _vehicle_to_sensor(cords, vehicle, sensor):
    """
    Transforms coordinates of a vehicle bounding box to sensor.
    """

    world_cord = _vehicle_to_world(cords, vehicle)
    sensor_cord = _world_to_sensor(world_cord, sensor)
    return sensor_cord

def _vehicle_to_world(cords, vehicle):
    """
    Transforms coordinates of a vehicle bounding box to world.
    """

    bb_transform = carla.Transform(vehicle.bounding_box.location)
    bb_vehicle_matrix = get_matrix(bb_transform)
    vehicle_world_matrix = get_matrix(vehicle.get_transform())
    bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
    world_cords = np.dot(bb_world_matrix, np.transpose(cords))
    return world_cords


def _world_to_sensor(cords, sensor):
    """
    Transforms world coordinates to sensor.
    """

    sensor_world_matrix = get_matrix(sensor.get_transform())
    world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
    sensor_cords = np.dot(world_sensor_matrix, cords)
    return sensor_cords
def get_matrix(transform):
    """
    Creates matrix from carla transform.
    """

    rotation = transform.rotation
    location = transform.location
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    c_r = np.cos(np.radians(rotation.roll))
    s_r = np.sin(np.radians(rotation.roll))
    c_p = np.cos(np.radians(rotation.pitch))
    s_p = np.sin(np.radians(rotation.pitch))
    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix
def draw_gt_boxes3d(gt_boxes3d, fi,  color=(1, 1, 1)):
    """
    Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (3,8) for XYZs of the box corners
        fig: figure handler
        color: RGB value tuple in range (0,1), box line color
    """
    bbox_plots = []
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        l = mlab.plot3d([gt_boxes3d[0, i], gt_boxes3d[0, j]], [gt_boxes3d[1, i], gt_boxes3d[1, j]],
                    [gt_boxes3d[2, i], gt_boxes3d[2, j]], tube_radius=None, line_width=2, color=color, reset_zoom=False, figure=fi)
        bbox_plots.append(l)

        i, j = k + 4, (k + 1) % 4 + 4
        l = mlab.plot3d([gt_boxes3d[0, i], gt_boxes3d[0, j]], [gt_boxes3d[1, i], gt_boxes3d[1, j]],
                    [gt_boxes3d[2, i], gt_boxes3d[2, j]], tube_radius=None, line_width=2, color=color, reset_zoom=False, figure=fi)
        bbox_plots.append(l)
        i, j = k, k + 4
        l = mlab.plot3d([gt_boxes3d[0, i], gt_boxes3d[0, j]], [gt_boxes3d[1, i], gt_boxes3d[1, j]],
                    [gt_boxes3d[2, i], gt_boxes3d[2, j]], tube_radius=None, line_width=2, color=color, reset_zoom=False, figure=fi)
        bbox_plots.append(l)
    return bbox_plots

def main(arg):
    """Main function of the script"""
    
    client = carla.Client(arg.host, arg.port)
    client.set_timeout(10.0)
    client.load_world('Town05')
    client.reload_world()
    world = client.get_world()
    try:
        original_settings = world.get_settings()
        settings = world.get_settings()
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

        delta = 0.05

        settings.fixed_delta_seconds = delta
        settings.synchronous_mode = True
        settings.no_rendering_mode = arg.no_rendering
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()
        # vehicle_bp = blueprint_library.filter(arg.filter)[0]
        # vehicle_transform = random.choice(world.get_map().get_spawn_points())
        # vehicle = world.spawn_actor(vehicle_bp, vehicle_transform)
        # vehicle.set_autopilot(arg.no_autopilot)
        lidar_loc = carla.Location(x=-65.0, y=3.0, z=6.0)
        lidar_bp = generate_lidar_bp(arg, world, blueprint_library, delta)
        lidar_transform = carla.Transform(lidar_loc)
        lidar = world.spawn_actor(lidar_bp, lidar_transform)

        fig = mlab.figure(size=(960,540), bgcolor=(0.05,0.05,0.05))
        vis = mlab.points3d(0, 0, 0, 0, mode='point', figure=fig, reset_zoom=False)
        buf = {'pts': np.zeros((1,3)), 'intensity':np.zeros(1)}
        mlab.view(distance=25)
        #  @mlab.animate(delay=100)
        def anim():
            i=0
            bbox_plots = []
            while True:
                veh_list = world.get_actors().filter('vehicle.*')
                veh_list = filter(lambda v: lidar_loc.distance(v.get_location()) < args.range, veh_list)
                veh_list = list(veh_list)
                for p in bbox_plots:
                    p.remove()
                # bbox_plots = []
                for v in veh_list:
                  print(v.get_transform())
                #     bb = _create_bb_points(v)
                #     bb = _vehicle_to_sensor(bb, v, lidar)[:3, :]
                #     p = draw_gt_boxes3d(bb, fig)
                #     bbox_plots += p
                vis.mlab_source.reset(x=buf['pts'][:,0], y=buf['pts'][:,1], z=buf['pts'][:,2], scalars=buf['intensity'])
                mlab.savefig(f'{i%10}.png', figure=fig)
                time.sleep(0.1)
                i+=1

        if arg.semantic:
            lidar.listen(lambda data: semantic_lidar_callback(data, buf))
        else:
            lidar.listen(lambda data: lidar_callback(data, buf))

        loopThread = threading.Thread(target=carlaEventLoop, args=[world], daemon=True)
        loopThread.start()
        anim()

    finally:
        world.apply_settings(original_settings)
        traffic_manager.set_synchronous_mode(False)
        lidar.destroy()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host CARLA Simulator (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port of CARLA Simulator (default: 2000)')
    argparser.add_argument(
        '--no-rendering',
        action='store_true',
        help='use the no-rendering mode which will provide some extra'
        ' performance but you will lose the articulated objects in the'
        ' lidar, such as pedestrians')
    argparser.add_argument(
        '--semantic',
        action='store_true',
        help='use the semantic lidar instead, which provides ground truth'
        ' information')
    argparser.add_argument(
        '--no-noise',
        action='store_true',
        help='remove the drop off and noise from the normal (non-semantic) lidar')
    argparser.add_argument(
        '--no-autopilot',
        action='store_false',
        help='disables the autopilot so the vehicle will remain stopped')
    argparser.add_argument(
        '--show-axis',
        action='store_true',
        help='show the cartesian coordinates axis')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='model3',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--upper-fov',
        default=15.0,
        type=float,
        help='lidar\'s upper field of view in degrees (default: 15.0)')
    argparser.add_argument(
        '--lower-fov',
        default=-45.0,
        type=float,
        help='lidar\'s lower field of view in degrees (default: -25.0)')
    argparser.add_argument(
        '--channels',
        default=64.0,
        type=float,
        help='lidar\'s channel count (default: 64)')
    argparser.add_argument(
        '--range',
        default=100.0,
        type=float,
        help='lidar\'s maximum range in meters (default: 100.0)')
    argparser.add_argument(
        '--points-per-second',
        default=500000,
        type=int,
        help='lidar\'s points per second (default: 500000)')
    argparser.add_argument(
        '-x',
        default=0.0,
        type=float,
        help='offset in the sensor position in the X-axis in meters (default: 0.0)')
    argparser.add_argument(
        '-y',
        default=0.0,
        type=float,
        help='offset in the sensor position in the Y-axis in meters (default: 0.0)')
    argparser.add_argument(
        '-z',
        default=0.0,
        type=float,
        help='offset in the sensor position in the Z-axis in meters (default: 0.0)')
    args = argparser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print(' - Exited by user.')
