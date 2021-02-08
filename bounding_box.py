try:
  import carla
except Exception as e:
  print('cannot import carla python library, please install into the python path now')
  exit(1)
import subprocess, os
import argparse
import time
import signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import numpy as np
from bb.carla_vehicle_annotator import auto_annotate, extract_depth
IM_HEIGHT = 1080
IM_WIDTH = 1920
camera = None
dcamera = None
PLT_IMG_HANDLER = None
PLT_IMG_AX = None

CURRENT_CAM_IMG = None
DEPTH_IMG = None
argparser = argparse.ArgumentParser(description=__doc__)
argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
argparser.add_argument('--map', default='Town05', type=str, help='choose map')
args = argparser.parse_args()

try:
  print('setting up client...')
  client = carla.Client(args.host, args.port)
  client.set_timeout(10.0)
  print('loading map')
  client.load_world(args.map)
  client.reload_world()
  print('loading world complete')
  # process = subprocess.Popen(['python','spawn_npc.py', '-n', '80', '--sync']) #spawn some npc
except Exception as e:
  print('error in creating the client, make sure your simulator is launched')
  print(str(e))
  # exit(1)
def processImg(image):
  global PLT_IMG_HANDLER
  global CURRENT_CAM_IMG
  i = np.array(image.raw_data)
  i = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
  i = i[:, :, :3]
  CURRENT_CAM_IMG = i
  # image.save_to_disk('image.png')
  return i/255.0
def processDepthImg(image):
  global DEPTH_IMG
  DEPTH_IMG = extract_depth(image)
  return DEPTH_IMG
def imgUpdate(i):
  global PLT_IMG_HANDLER
  global PLT_IMG_AX
  actors = world.get_actors()
  vehicles = actors.filter('vehicle.*')
  walkers = actors.filter('walker.*')
  # print (vehicles)
  if not PLT_IMG_HANDLER:
    PLT_IMG_HANDLER = plt.imshow(CURRENT_CAM_IMG)
    #plt.show()
  else:
    PLT_IMG_HANDLER.set_data(CURRENT_CAM_IMG)
  vs, removed = auto_annotate(vehicles, camera, DEPTH_IMG)
  [p.remove() for p in reversed(PLT_IMG_AX.patches)] #clear previous bounding boxes on the figure
  for box in vs['bbox']:
    # print (box)
    rect = patches.Rectangle(
      (int(box[0][0]),int(box[0][1])),
      int(box[1][0]-box[0][0]), #width
      int(box[1][1] - box[0][1]), #height
      linewidth=1,edgecolor='r',facecolor='none')
    PLT_IMG_AX.add_patch(rect)
print('successfully connected to Carla')
world = client.get_world()
settings = world.get_settings()
world.apply_settings(settings)
map = world.get_map()
bp = world.get_blueprint_library().find('sensor.camera.rgb')
bp.set_attribute('image_size_x', str(IM_WIDTH))
bp.set_attribute('image_size_y', str(IM_HEIGHT))
bp.set_attribute('fov', '110')
# Set the time in seconds between sensor captures
bp.set_attribute('sensor_tick', '0.2')
transform = carla.Transform(carla.Location(x=-65.0, y=3.0, z=6.0), carla.Rotation(yaw=180.0, pitch=-30.0))
camera = world.spawn_actor(bp, transform)
camera.listen(processImg)
dbp = world.get_blueprint_library().find('sensor.camera.depth')
dbp.set_attribute('image_size_x', str(IM_WIDTH))
dbp.set_attribute('image_size_y', str(IM_HEIGHT))
dbp.set_attribute('fov', '110')
dcamera = world.spawn_actor(dbp, transform)
dcamera.listen(processDepthImg)
_, PLT_IMG_AX = plt.subplots(1)
ani = FuncAnimation(plt.gcf(), imgUpdate, interval=50)
plt.show()

# image.raw_data

# # Default format (depends on the camera PostProcessing but always a numpy array).
# image.data

# # numpy BGRA array.
# image_converter.to_bgra_array(image)

# # numpy RGB array.
# image_converter.to_rgb_array(image)
time.sleep(2)
while True:
  world.wait_for_tick()