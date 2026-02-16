# pip install retina-face
# we recommand tensorflow==2.15
import os
import logging

# Prevent TensorFlow from grabbing all GPU memory
# We must do this before importing retinaface/tensorflow
try:
    import tensorflow as tf
    # Hide GPUs from TensorFlow so it uses CPU only
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except ImportError:
    pass
except Exception as e:
    logging.warning(f"[FaceDetect] Failed to hide GPU from TensorFlow: {e}")

from retinaface import RetinaFace
import sys
from PIL import Image
import numpy as np

def get_mask_coord(image_path):

  img = Image.open(image_path).convert("RGB")
  img = np.array(img)[:,:,::-1]
  if img is None:
    raise ValueError(f"Exception while loading {img}")

  height, width, _ = img.shape

  facial_areas = resp = RetinaFace.detect_faces(img) 
  if len(facial_areas) == 0:
    print (f'{image_path} has no face detected!')
    return None
  else:
    face = facial_areas['face_1']
    x,y,x2,y2 = face["facial_area"]
    
    return y,y2,x,x2,height,width

if __name__ == "__main__":
  image_path = sys.argv[1]
  y,y2,x,x2,height,width = get_mask_coord(image_path)
  print (y,y2,x,x2,height,width)