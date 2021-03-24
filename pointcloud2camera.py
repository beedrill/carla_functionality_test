#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Lidar projection on RGB camera example
"""

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import argparse
from queue import Queue
from queue import Empty
from matplotlib import cm
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    from PIL import Image
except ImportError:
    raise RuntimeError('cannot import PIL, make sure "Pillow" package is installed')

VIRIDIS = np.array(cm.get_cmap('viridis').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
original_settings = None
delta = 0.05

def generate_lidar_bp(arg, world, blueprint_library, delta):
    """Generates a CARLA blueprint based on the script parameters"""
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    # if arg.no_noise:
    lidar_bp.set_attribute('dropoff_general_rate', '0.0')
    lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
    lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
    # else:
    #     lidar_bp.set_attribute('noise_stddev', '0.05')

    lidar_bp.set_attribute('upper_fov', str(arg.upper_fov))
    lidar_bp.set_attribute('lower_fov', str(arg.lower_fov))
    lidar_bp.set_attribute('channels', str(arg.channels))
    lidar_bp.set_attribute('range', str(arg.range))
    lidar_bp.set_attribute('rotation_frequency', str(1.0 / delta))
    lidar_bp.set_attribute('points_per_second', str(arg.points_per_second))
    return lidar_bp

def generate_camera_bp(arg, blueprint_library):
    camera_bp = blueprint_library.filter("sensor.camera.rgb")[0]
    camera_bp.set_attribute("image_size_x", str(arg.width))
    camera_bp.set_attribute("image_size_y", str(arg.height))
    camera_bp.set_attribute('fov', '110')
    return camera_bp

def sensor_callback(data, queue):
    """
    This simple callback just stores the data on a thread safe Python Queue
    to be retrieved from the "main thread".
    """
    queue.put(data)

def setup_world(args):
    global original_settings
    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)
    client.load_world('town05')
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = delta
    world.apply_settings(settings)
    lidar_bp = generate_lidar_bp(args, world, bp_lib, delta)
    lidar_transform = carla.Transform(carla.Location(x=-65.0, y=3.0, z=6.0))
    lidar = world.spawn_actor(
        blueprint=lidar_bp,
        transform=lidar_transform)
    camera_bp = generate_camera_bp(args, bp_lib)
    camera_transform = transform = carla.Transform(carla.Location(x=-65.0, y=3.0, z=6.0), carla.Rotation(yaw=180.0, pitch=-30.0))
    camera = world.spawn_actor(
        blueprint=camera_bp,
        transform=camera_transform)
    return client, camera, lidar, camera_bp, lidar_bp, world
def transform_polar (points):
    x = points[:, 0]
    y = points[:, 1]
    rho = np.sqrt(x**2 + y**2)
    phi = np.degrees(np.arctan2(y,x))
    return np.vstack((rho, phi)).T  

def lidar2cam(args, camera, lidar, camera_bp, world, background_data):
    """
    This function is intended to be a tutorial on how to retrieve data in a
    synchronous way, and project 3D points from a lidar to a 2D camera.
    """
    try:
        # Build the K projection matrix:
        # K = [[Fx,  0, image_w/2],
        #      [ 0, Fy, image_h/2],
        #      [ 0,  0,         1]]
        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov").as_float()
        focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))

        # In this case Fx and Fy are the same since the pixel aspect
        # ratio is 1
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = image_w / 2.0
        K[1, 2] = image_h / 2.0
        # print(f'K={K}')
        # The sensor data will be saved in thread-safe Queues
        image_queue = Queue()
        lidar_queue = Queue()

        camera.listen(lambda data: sensor_callback(data, image_queue))
        lidar.listen(lambda data: sensor_callback(data, lidar_queue))

        for frame in range(args.frames):
            world.tick()
            world_frame = world.get_snapshot().frame

            try:
                # Get the data once it's received.
                image_data = image_queue.get(True, 1.0)
                lidar_data = lidar_queue.get(True, 1.0)
            except Empty:
                print("[Warning] Some sensor data has been missed")
                continue
            print(f'lidar frame: {lidar_data.frame}, image frame: {image_data.frame}, world frame: {world_frame}')
            # assert image_data.frame == lidar_data.frame == world_frame
            # At this point, we have the synchronized information from the 2 sensors.
            sys.stdout.write("\r(%d/%d) Simulation: %d Camera: %d Lidar: %d" %
                (frame, args.frames, world_frame, image_data.frame, lidar_data.frame) + ' \n')
            # sys.stdout.flush()

            # Get the raw BGRA buffer and convert it to an array of RGB of
            # shape (image_data.height, image_data.width, 3).
            im_array = np.copy(np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8")))
            im_array = np.reshape(im_array, (image_data.height, image_data.width, 4))
            im_array = im_array[:, :, :3][:, :, ::-1]

            # Get the lidar data and convert it to a numpy array.
            p_cloud_size = len(lidar_data)
            p_cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
            p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))

            # Lidar intensity array of shape (p_cloud_size,) but, for now, let's
            # focus on the 3D points.
            intensity = np.array(p_cloud[:, 3])

            # Point cloud in lidar sensor space array of shape (3, p_cloud_size).
            local_lidar_points = np.array(p_cloud[:, :-1]).T
            local_lidar_points[:, :1] = -local_lidar_points[:, :1]
            # print(local_lidar_points)
            # print(points.shape)
            # d, i = background_data.query(points)
            # filtermask = d > 0.2
            # filtered_points = points[filtermask]
            # polar_points = transform_polar(filtered_points)
            # polar_points = np.append(polar_points,np.expand_dims(filtered_points[:,2], axis=1), 1)
            # polar_points[:, 1] *= 0.05
            # polar_points[:, 2] *= 0.6
            # if points[filtermask].shape[0] > 0:
            #     local_lidar_points = (np.array(p_cloud[:, :3])[filtermask]).T
            #     intensity = intensity[filtermask]
            # else:
            #     local_lidar_points = np.array([[0], [0], [0]])
            #     intensity = np.array([0])

            # Add an extra 1.0 at the end of each 3d point so it becomes of
            # shape (4, p_cloud_size) and it can be multiplied by a (4, 4) matrix.
            local_lidar_points = np.r_[
                local_lidar_points, [np.ones(local_lidar_points.shape[1])]]
            # print(local_lidar_points.shape)
            # This (4, 4) matrix transforms the points from lidar space to world space.
            lidar_2_world = lidar.get_transform().get_matrix()
            # print(lidar_2_world)
            # Transform the points from lidar space to world space.
            world_points = np.dot(lidar_2_world, local_lidar_points)
            # print(world_points)
            # This (4, 4) matrix transforms the points from world to sensor coordinates.
            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

            # Transform the points from world space to camera space.
            sensor_points = np.dot(world_2_camera, world_points)

            # Now we must change from UE4's coordinate system to an "standard"
            # camera coordinate system (the same used by OpenCV):

            # ^ z                       . z
            # |                        /
            # |              to:      +-------> x
            # | . x                   |
            # |/                      |
            # +-------> y             v y

            # This can be achieved by multiplying by the following matrix:
            # [[ 0,  1,  0 ],
            #  [ 0,  0, -1 ],
            #  [ 1,  0,  0 ]]

            # Or, in this case, is the same as swapping:
            # (x, y ,z) -> (y, -z, x)
            point_in_camera_coords = np.array([
                sensor_points[1],
                sensor_points[2] * -1,
                sensor_points[0]])

            # Finally we can use our K matrix to do the actual 3D -> 2D.
            points_2d = np.dot(K, point_in_camera_coords)

            # Remember to normalize the x, y values by the 3rd value.
            points_2d = np.array([
                points_2d[0, :] / points_2d[2, :],
                points_2d[1, :] / points_2d[2, :],
                points_2d[2, :]])

            # At this point, points_2d[0, :] contains all the x and points_2d[1, :]
            # contains all the y values of our points. In order to properly
            # visualize everything on a screen, the points that are out of the screen
            # must be discarted, the same with points behind the camera projection plane.
            points_2d = points_2d.T
            intensity = intensity.T
            points_in_canvas_mask = \
                (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < image_w) & \
                (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < image_h) & \
                (points_2d[:, 2] > 0.0)
            points_2d = points_2d[points_in_canvas_mask]
            intensity = intensity[points_in_canvas_mask]

            # Extract the screen coords (uv) as integers.
            u_coord = points_2d[:, 0].astype(np.int)
            v_coord = points_2d[:, 1].astype(np.int)

            # Since at the time of the creation of this script, the intensity function
            # is returning high values, these are adjusted to be nicely visualized.
            intensity = 4 * intensity - 3
            color_map = np.array([
                np.interp(intensity, VID_RANGE, VIRIDIS[:, 0]) * 255.0,
                np.interp(intensity, VID_RANGE, VIRIDIS[:, 1]) * 255.0,
                np.interp(intensity, VID_RANGE, VIRIDIS[:, 2]) * 255.0]).astype(np.int).T

            if args.dot_extent <= 0:
                # Draw the 2d points on the image as a single pixel using numpy.
                im_array[v_coord, u_coord] = color_map
            else:
                # Draw the 2d points on the image as squares of extent args.dot_extent.
                for i in range(len(points_2d)):
                    # I'm not a NumPy expert and I don't know how to set bigger dots
                    # without using this loop, so if anyone has a better solution,
                    # make sure to update this script. Meanwhile, it's fast enough :)
                    im_array[
                        v_coord[i]-args.dot_extent : v_coord[i]+args.dot_extent,
                        u_coord[i]-args.dot_extent : u_coord[i]+args.dot_extent] = color_map[i]

            # Save the image using Pillow module.
            image = Image.fromarray(im_array)
            image.save("_out/%08d.png" % image_data.frame)

    finally:
        # Apply the original settings when exiting.
        world.apply_settings(original_settings)

        # Destroy the actors in the scene.
        if camera:
            camera.destroy()
        if lidar:
            lidar.destroy()
def process_background(data):
    # points = data[:, :-1]
    background_data = cKDTree(data)
    return background_data
def main(args):

    args.width, args.height = [int(x) for x in args.res.split('x')]
    args.dot_extent -= 1
    data = np.genfromtxt('background.csv', delimiter=',')
    print('loading background data, size: {}'.format(data.shape))
    background_data = process_background(data)

    try:
        client, camera, lidar, camera_bp, lidar_bp, world = setup_world(args)
        lidar2cam(args, camera, lidar, camera_bp, world, background_data)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    """Start function"""
    argparser = argparse.ArgumentParser(
        description='CARLA Sensor sync and projection tutorial')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1920x1080',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '-f', '--frames',
        metavar='N',
        default=500,
        type=int,
        help='number of frames to record (default: 500)')
    argparser.add_argument(
        '-d', '--dot-extent',
        metavar='SIZE',
        default=4,
        type=int,
        help='visualization dot extent in pixels (Recomended [1-4]) (default: 2)')
    argparser.add_argument(
        '--no-noise',
        action='store_true',
        help='remove the drop off and noise from the normal (non-semantic) lidar')
    argparser.add_argument(
        '--upper-fov',
        metavar='F',
        default=15.0,
        type=float,
        help='lidar\'s upper field of view in degrees (default: 15.0)')
    argparser.add_argument(
        '--lower-fov',
        metavar='F',
        default=-85.0,
        type=float,
        help='lidar\'s lower field of view in degrees (default: -25.0)')
    argparser.add_argument(
        '-c', '--channels',
        metavar='C',
        default=160.0,
        type=float,
        help='lidar\'s channel count (default: 64)')
    argparser.add_argument(
        '-r', '--range',
        metavar='R',
        default=100.0,
        type=float,
        help='lidar\'s maximum range in meters (default: 100.0)')
    argparser.add_argument(
        '--points-per-second',
        metavar='N',
        default='1000000',
        type=int,
        help='lidar points per second (default: 1000000)')
    args = argparser.parse_args()
    main(args)
