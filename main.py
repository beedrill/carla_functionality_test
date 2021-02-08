try:
  import carla
except Exception as e:
  print('cannot import carla python library, please install into the python path now')
  exit(1)
import time
import subprocess, os
try:
  print('setting up client...')
  client = carla.Client('127.0.0.1', 2000)
  client.set_timeout(10.0)
  print('loading map')
  client.load_world('Town05')
  client.reload_world()
  print('loading world complete')
  # process = subprocess.Popen(['python','spawn_npc.py', '-n', '80', '--sync']) #spawn some npc
except Exception as e:
  print('error in creating the client, make sure your simulator is launched')
  print(str(e))
  exit(1)

IM_HEIGHT = 1080
IM_WIDTH = 1920
RESULT_FOLDER_PATH = 'results/'

print('successfully connected to Carla')

world = client.get_world()
settings = world.get_settings()
delta = 0.05
settings.fixed_delta_seconds = delta
world.apply_settings(settings)
map = world.get_map()
### camera settigns ###
rgb_cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
rgb_cam_bp.set_attribute('image_size_x', str(IM_WIDTH))
rgb_cam_bp.set_attribute('image_size_y', str(IM_HEIGHT))
rgb_cam_bp.set_attribute('fov', '110')
# bp.set_attribute('sensor_tick', '5.0')
transform = carla.Transform(carla.Location(x=-65.0, y=3.0, z=6.0), carla.Rotation(yaw=180.0, pitch=-30.0))
camera = world.spawn_actor(rgb_cam_bp, transform)
camera.listen(lambda image: image.save_to_disk('output/camera%06d.png' % image.frame))
### lidar settings ###
# lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
# lidar_bp.set_attribute('rotation_frequency', str(1.0 / delta))
# lidar_bp.set_attribute('sensor_tick', '5.0')
# transform = carla.Transform(carla.Location(x=-65.0, y=3.0, z=6.0))
# lidar = world.spawn_actor(lidar_bp, transform)
# def lidar_callback(d):
#   d.save_to_disk('output/lidar%06d.ply' % d.frame)
# lidar.listen(lambda data: lidar_callback(data))
process = subprocess.Popen(['python','spawn_npc.py', '-n', '20']) #spawn some npc
time.sleep(2)
while True:
  world.wait_for_tick()