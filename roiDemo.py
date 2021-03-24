import carla
from queue import Queue
from queue import Empty
import numpy as np
import sys
import glob
import os
from matplotlib import cm

try:
    from PIL import Image
except ImportError:
    raise RuntimeError('cannot import PIL, make sure "Pillow" package is installed')
global original_settings
IMG_WIDTH = 1920
IMG_HEIGHT = 1080
CAMERA_FOV = 110
delta = 0.05
UPPER_FOV = 15.0
LOWER_FOV = -85.0
CHANNELS = 160.0
RANGE = 100.0
POINTS_PER_SECOND = 1000000
FRAMES = 500
image_queue = Queue()
lidar_queue = Queue()
VIRIDIS = np.array(cm.get_cmap('viridis').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
LIDAR_TYPE = 'semantic'
ONLY_VEHICLES = True
LABEL_COLORS = np.array([
    (255, 255, 255), # None
    (70, 70, 70),    # Building
    (100, 40, 40),   # Fences
    (55, 90, 80),    # Other
    (220, 20, 60),   # Pedestrian
    (153, 153, 153), # Pole
    (157, 234, 50),  # RoadLines
    (128, 64, 128),  # Road
    (244, 35, 232),  # Sidewalk
    (107, 142, 35),  # Vegetation
    (0, 0, 142),     # Vehicle
    (102, 102, 156), # Wall
    (220, 220, 0),   # TrafficSign
    (70, 130, 180),  # Sky
    (81, 0, 81),     # Ground
    (150, 100, 100), # Bridge
    (230, 150, 140), # RailTrack
    (180, 165, 180), # GuardRail
    (250, 170, 30),  # TrafficLight
    (110, 190, 160), # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),   # Water
    (145, 170, 100), # Terrain
]) # normalize each channel [0-1] since is what Open3D uses
def sensor_callback(data, queue):
    """
    This simple callback just stores the data on a thread safe Python Queue
    to be retrieved from the "main thread".
    """
    queue.put(data)

def generate_lidar_bp(world, blueprint_library):
    """Generates a CARLA blueprint based on the script parameters"""
    # lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    if LIDAR_TYPE == 'semantic':
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast_semantic')
    else:
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('dropoff_general_rate', '0.0')
        lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
        lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')

    lidar_bp.set_attribute('upper_fov', str(UPPER_FOV))
    lidar_bp.set_attribute('lower_fov', str(LOWER_FOV))
    lidar_bp.set_attribute('channels', str(CHANNELS))
    lidar_bp.set_attribute('range', str(RANGE))
    lidar_bp.set_attribute('rotation_frequency', str(1.0 / delta))
    lidar_bp.set_attribute('points_per_second', str(POINTS_PER_SECOND))
    return lidar_bp

def generate_camera_bp(blueprint_library):
    camera_bp = blueprint_library.filter("sensor.camera.rgb")[0]
    camera_bp.set_attribute("image_size_x", str(IMG_WIDTH))
    camera_bp.set_attribute("image_size_y", str(IMG_HEIGHT))
    camera_bp.set_attribute('fov', str(CAMERA_FOV))
    return camera_bp
def draw_img_with_points(points_2d, intensity, im_array):
    dot_extent = 2
    # Extract the screen coords (uv) as integers.
    u_coord = points_2d[:, 0].astype(np.int)
    v_coord = points_2d[:, 1].astype(np.int)
    # Since at the time of the creation of this script, the intensity function
    # is returning high values, these are adjusted to be nicely visualized.
    # color_map = np.array([
    #     np.interp(intensity, VID_RANGE, VIRIDIS[:, 0]) * 255.0,
    #     np.interp(intensity, VID_RANGE, VIRIDIS[:, 1]) * 255.0,
    #     np.interp(intensity, VID_RANGE, VIRIDIS[:, 2]) * 255.0]).astype(np.int).T
    if LIDAR_TYPE == 'semantic':
        color_map = LABEL_COLORS[intensity]

    else:
        intensity = 4 * intensity - 3
        color_map = np.array([
            np.interp(intensity, VID_RANGE, VIRIDIS[:, 0]) * 255.0,
            np.interp(intensity, VID_RANGE, VIRIDIS[:, 1]) * 255.0,
            np.interp(intensity, VID_RANGE, VIRIDIS[:, 2]) * 255.0]).astype(np.int).T
    if dot_extent <= 0:
        # Draw the 2d points on the image as a single pixel using numpy.
        im_array[v_coord, u_coord] = color_map
    else:
        # Draw the 2d points on the image as squares of extent dot_extent.
        for i in range(len(points_2d)):
            # I'm not a NumPy expert and I don't know how to set bigger dots
            # without using this loop, so if anyone has a better solution,
            # make sure to update this script. Meanwhile, it's fast enough :)
            im_array[
                v_coord[i]-dot_extent : v_coord[i]+dot_extent,
                u_coord[i]-dot_extent : u_coord[i]+dot_extent] = color_map[i]

    # Save the image using Pillow module.
    image = Image.fromarray(im_array)
    return image

def lidar2cam(image_data, lidar_data, camera, lidar, camera_bp):
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()
    fov = camera_bp.get_attribute("fov").as_float()
    focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))
    # Build the K projection matrix:
    # K = [[Fx,  0, image_w/2],
    #      [ 0, Fy, image_h/2],
    #      [ 0,  0,         1]]
    # In this case Fx and Fy are the same since the pixel aspect
    # ratio is 1
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = image_w / 2.0
    K[1, 2] = image_h / 2.0
    # 1 print(f'K={K}')
    # Get the raw BGRA buffer and convert it to an array of RGB of
    # shape (image_data.height, image_data.width, 3).
    im_array = np.copy(np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8")))
    im_array = np.reshape(im_array, (image_data.height, image_data.width, 4))
    im_array = im_array[:, :, :3][:, :, ::-1]

    # Get the lidar data and convert it to a numpy array.
    if LIDAR_TYPE == 'semantic':
        p_cloud = np.frombuffer(lidar_data.raw_data, dtype=np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))
        # Point cloud in lidar sensor space array of shape (3, p_cloud_size).
        local_lidar_points = np.array([p_cloud['x'], p_cloud['y'], p_cloud['z']])
        intensity = np.array(p_cloud['ObjTag'])
        
    else:
        p_cloud_size = len(lidar_data)
        p_cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
        p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))
        intensity = np.array(p_cloud[:, 3])
        # Point cloud in lidar sensor space array of shape (3, p_cloud_size).
        local_lidar_points = np.array(p_cloud[:, :-1]).T
        local_lidar_points[:, :1] = -local_lidar_points[:, :1]
    # Lidar intensity array of shape (p_cloud_size,) but, for now, let's
    # focus on the 3D points.
    # print(local_lidar_points)
    # Add an extra 1.0 at the end of each 3d point so it becomes of
    # shape (4, p_cloud_size) and it can be multiplied by a (4, 4) matrix.
    local_lidar_points = np.r_[
        local_lidar_points, [np.ones(local_lidar_points.shape[1])]]
    # This (4, 4) matrix transforms the points from lidar space to world space.
    # print(local_lidar_points)
    # print(local_lidar_points.shape)
    lidar_2_world = lidar.get_transform().get_matrix()
    # print(f'lidar2world: {lidar_2_world}')
    # Transform the points from lidar space to world space.
    world_points = np.dot(lidar_2_world, local_lidar_points)
    # print(world_points)
    # This (4, 4) matrix transforms the points from world to sensor coordinates.
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
    # Transform the points from world space to camera space.
    sensor_points = np.dot(world_2_camera, world_points)
    # Nww we must change from UE4's coordinate system to an "standard"
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
    # print(points_2d)
    intensity = intensity[points_in_canvas_mask]
    # print(intensity)
    return points_2d, intensity, im_array

def setup_world():
    global original_settings
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(2.0)
    client.load_world('town05')
    client.reload_world()
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = delta
    world.apply_settings(settings)
    camera_bp = generate_camera_bp(bp_lib)
    camera_transform = carla.Transform(carla.Location(x=-65.0, y=3.0, z=6.0), carla.Rotation(yaw=180.0, pitch=-30.0))
    camera = world.spawn_actor(
        blueprint=camera_bp,
        transform=camera_transform)
    lidar_bp = generate_lidar_bp(world, bp_lib)
    lidar_transform = carla.Transform(carla.Location(x=-65.0, y=3.0, z=6.0))
    #lidar = world.spawn_actor(lidar_bp, lidar_transform)
    lidar = world.spawn_actor(
        blueprint=lidar_bp,
        transform=lidar_transform)
    # The sensor data will be saved in thread-safe Queues
    camera.listen(lambda data: sensor_callback(data, image_queue))
    lidar.listen(lambda data: sensor_callback(data, lidar_queue))
    return client, camera, lidar, camera_bp, lidar_bp, world
def main():
    client, camera, lidar, camera_bp, lidar_bp, world = setup_world()
    try:
        for frame in range(FRAMES):
            world.tick()
            world_frame = world.get_snapshot().frame
            try:
                image_data = image_queue.get(True, 1.0)
                lidar_data = lidar_queue.get(True, 1.0)
            except Empty:
                print("[Warning] Some sensor data has been missed")
                continue
            # At this point, we have the synchronized information from the 2 sensors.
            sys.stdout.write("\r(%d/%d) Simulation: %d Camera: %d Lidar: %d" %
                (frame, FRAMES, world_frame, image_data.frame, lidar_data.frame) + ' \n')
            print(f'lidar frame: {lidar_data.frame}, image frame: {image_data.frame}, world frame: {world_frame}')
            points_2d, intensity, im_array = lidar2cam(image_data, lidar_data, camera, lidar, camera_bp)
            new_img = draw_img_with_points(points_2d, intensity, im_array)
            new_img.save("_out/%08d.png" % image_data.frame)
    finally:
        world.apply_settings(original_settings)
        if camera:
            camera.destroy()
        if lidar:
            lidar.destroy()

if __name__ == '__main__':
    main()