# Copyright (C) 2024 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials,
# and your use of them is governed by the express license under which they
# were provided to you ("License"). Unless the License provides otherwise,
# you may not use, modify, copy, publish, distribute, disclose or transmit
# this software or the related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with no express
# or implied warranties, other than those that are expressly stated in the License.

import base64
import json
import logging
import math
import os
import struct
import time
from collections import defaultdict
from datetime import datetime
from uuid import getnode as get_mac
import cv2
import ntplib
import numpy as np
import paho.mqtt.client as mqtt
from pytz import timezone
from utils import publisher_utils as utils

ROOT_CA = os.environ.get('ROOT_CA', '/run/secrets/certs/scenescape-ca.pem')
CLIENT_CERT = os.environ.get('CLIENT_CERT', '/run/secrets/certs/scenescape-broker.crt')
CLIENT_KEY = os.environ.get('CLIENT_KEY', '/run/secrets/certs/scenescape-broker.key')
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
TIMEZONE = "UTC"

COLOR_LABELS = ["black", "silver", "white", "red", "blue", "gray"]

def getMACAddress():
  if 'MACADDR' in os.environ:
    return os.environ['MACADDR']
  a = get_mac()
  h = iter(hex(a)[2:].zfill(12))
  return ":".join(i + next(h) for i in h)

class PostDecodeTimestampCapture:
  def __init__(self, ntpServer=None):
    self.log = logging.getLogger('SSCAPE_ADAPTER')
    self.log.setLevel(logging.INFO)
    self.ntpClient = ntplib.NTPClient()
    self.ntpServer = ntpServer
    self.lastTimeSync = None
    self.timeOffset = 0
    self.ts = None
    self.timestamp_for_next_block = None
    self.fps = 5.0
    self.fps_alpha = 0.75
    self.last_calculated_fps_ts = None
    self.fps_calc_interval = 1
    self.frame_cnt = 0

  def processFrame(self, frame):
    now = time.time()
    self.frame_cnt += 1
    if not self.last_calculated_fps_ts:
      self.last_calculated_fps_ts = now
    if (now - self.last_calculated_fps_ts) > self.fps_calc_interval:
      self.fps = self.fps * self.fps_alpha + (1 - self.fps_alpha) * (self.frame_cnt / (now - self.last_calculated_fps_ts))
      self.last_calculated_fps_ts = now
      self.frame_cnt = 0
    if self.ntpServer:
      if not self.lastTimeSync or now - self.lastTimeSync > 1000 :
        try:
          response = self.ntpClient.request(host=self.ntpServer, port=123)
          self.timeOffset = response.offset
          self.lastTimeSync = now
        except Exception:
          pass
    now += self.timeOffset
    self.timestamp_for_next_block = now
    frame.add_message(json.dumps({
      'postdecode_timestamp': f"{datetime.fromtimestamp(now, tz=timezone(TIMEZONE)).strftime(DATETIME_FORMAT)[:-3]}Z",
      'timestamp_for_next_block': now,
      'fps': self.fps
    }))
    return True

def computeObjBoundingBoxParams(pobj, fw, fh, x, y, w, h, xminnorm=None, yminnorm=None, xmaxnorm=None, ymaxnorm=None):
  xmax, xmin = int(xmaxnorm * fw), int(xminnorm * fw)
  ymax, ymin = int(ymaxnorm * fh), int(yminnorm * fh)
  comw, comh = (xmax - xmin) / 3, (ymax - ymin) / 4
  pobj.update({
    'center_of_mass': {'x': int(xmin + comw), 'y': int(ymin + comh), 'width': comw, 'height': comh},
    'bounding_box_px': {'x': x, 'y': y, 'width': w, 'height': h}
  })
  return

def detectionPolicy(pobj, item, fw, fh):
  pobj.update({
    'category': item['detection']['label'],
    'confidence': item['detection']['confidence']
  })
  computeObjBoundingBoxParams(pobj, fw, fh, item['x'], item['y'], item['w'], item['h'],
                              item['detection']['bounding_box']['x_min'],
                              item['detection']['bounding_box']['y_min'],
                              item['detection']['bounding_box']['x_max'],
                              item['detection']['bounding_box']['y_max'])
  return

def reidPolicy(pobj, item, fw, fh):
  detectionPolicy(pobj, item, fw, fh)
  reid_vector = item['tensors'][1]['data']
  v = struct.pack("256f", *reid_vector)
  pobj['reid'] = base64.b64encode(v).decode('utf-8')
  return

def classificationPolicy(pobj, item, fw, fh):
  detectionPolicy(pobj, item, fw, fh)
  for k, v in item.items():
    if k.startswith('classification_layer_name:'):
      pobj['category'] = v.get('label', pobj['category'])
      break
  return

metadatapolicies = {
  "detectionPolicy": detectionPolicy,
  "reidPolicy": reidPolicy,
  "classificationPolicy": classificationPolicy
}

class PostInferenceDataPublish:
  def __init__(self, cameraid, metadatagenpolicy='detectionPolicy', publish_image=False):
    self.cameraid = cameraid
    self.is_publish_image = publish_image
    self.is_publish_calibration_image = False
    self.start_time = time.time()
    
    # Topic derivation: parkingcam1_gpu -> object_detection_1
    cam_num = ''.join(filter(str.isdigit, self.cameraid)) or '1'
    self.sp_topic = f"object_detection_{cam_num}"
    
    self.setupMQTT()
    self.metadatagenpolicy = metadatapolicies.get(metadatagenpolicy, detectionPolicy)
    self.frame_level_data = {'id': cameraid, 'debug_mac': getMACAddress()}
    self.frame_count = 0
    return

  def on_connect(self, client, userdata, flags, rc):
    if rc == 0:
      self.client.subscribe(f"scenescape/cmd/camera/{self.cameraid}")
    return

  def setupMQTT(self):
    self.client = mqtt.Client()
    self.client.on_connect = self.on_connect
    self.broker = "broker.scenescape.intel.com"
    if ROOT_CA and os.path.exists(ROOT_CA):
      if os.path.exists(CLIENT_CERT) and os.path.exists(CLIENT_KEY):
        self.client.tls_set(ca_certs=ROOT_CA, certfile=CLIENT_CERT, keyfile=CLIENT_KEY)
      else:
        self.client.tls_set(ca_certs=ROOT_CA)
      self.client.tls_insecure_set(True)
    self.client.connect(self.broker, 1883, 120)
    self.client.on_message = self.handleCameraMessage
    self.client.loop_start()
    return

  def handleCameraMessage(self, client, userdata, message):
    msg = str(message.payload.decode("utf-8"))
    if msg == "getimage":
      self.is_publish_image = True
    elif msg == "getcalibrationimage":
      self.is_publish_calibration_image = True
    return

  def annotateObjects(self, img):
    objColors = ((0, 0, 255), (255, 128, 128), (207, 83, 294), (31, 156, 238))
    for otype, objects in self.frame_level_data['objects'].items():
      cindex = 1 if otype in ("vehicle", "car", "bicycle") else 0
      for obj in objects:
        topleft_cv = (int(obj['bounding_box_px']['x']), int(obj['bounding_box_px']['y']))
        bottomright_cv = (int(obj['bounding_box_px']['x'] + obj['bounding_box_px']['width']),
                          int(obj['bounding_box_px']['y'] + obj['bounding_box_px']['height']))
        cv2.rectangle(img, topleft_cv, bottomright_cv, objColors[cindex], 4)
    return

  def annotateFPS(self, img, fpsval):
    fpsStr = f'FPS {fpsval:.1f}'
    scale = int((img.shape[0] + 479) / 480)
    cv2.putText(img, fpsStr, (0, 30 * scale), cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, (0,0,0), 5 * scale)
    cv2.putText(img, fpsStr, (0, 30 * scale), cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, (255,255,255), 2 * scale)
    return

  def buildImgData(self, imgdatadict, gvaframe, annotate):
    imgdatadict.update({
      'timestamp': self.frame_level_data['timestamp'],
      'id': self.cameraid
    })
    with gvaframe.data() as image:
      if annotate:
        self.annotateObjects(image)
        self.annotateFPS(image, self.frame_level_data['rate'])
      _, jpeg = cv2.imencode(".jpg", image)
    imgdatadict['image'] = base64.b64encode(jpeg).decode('utf-8')
    return

  def buildObjData(self, gvadata):
    now = time.time()
    self.frame_level_data.update({
      'timestamp': gvadata.get('postdecode_timestamp', datetime.utcnow().isoformat()),
      'debug_timestamp_end': f"{datetime.fromtimestamp(now, tz=timezone(TIMEZONE)).strftime(DATETIME_FORMAT)[:-3]}Z",
      'debug_processing_time': now - float(gvadata.get('timestamp_for_next_block', now)),
      'rate': float(gvadata.get('fps', 30.0))
    })
    objects = defaultdict(list)
    if 'objects' in gvadata and len(gvadata['objects']) > 0:
      framewidth = gvadata.get('resolution', {}).get('width', 1920)
      frameheight = gvadata.get('resolution', {}).get('height', 1080)
      for det in gvadata['objects']:
        vaobj = {}
        self.metadatagenpolicy(vaobj, det, framewidth, frameheight)
        otype = vaobj['category']
        vaobj['id'] = len(objects[otype]) + 1
        objects[otype].append(vaobj)
    self.frame_level_data['objects'] = objects

  def processFrame(self, frame):
    if self.client.is_connected():
      self.frame_count += 1
      now_ns = time.time_ns()
      gvametadata, imgdatadict = {}, {}
      
      # Fast C++ metadata extraction
      utils.get_gva_meta_messages(frame, gvametadata)
      gvametadata['gva_meta'] = utils.get_gva_meta_regions(frame)
      self.buildObjData(gvametadata)

      # 1. Image Snapshots for SceneScape UI
      if self.is_publish_image:
        self.buildImgData(imgdatadict, frame, True)
        self.client.publish(f"scenescape/image/camera/{self.cameraid}", json.dumps(imgdatadict))
        self.is_publish_image = False
      if self.is_publish_calibration_image:
        if not imgdatadict:
          self.buildImgData(imgdatadict, frame, False)
        self.client.publish(f"scenescape/image/calibration/camera/{self.cameraid}", json.dumps(imgdatadict))
        self.is_publish_calibration_image = False

      # 2. Publish SceneScape 2D/3D Camera Sensor Payload
      self.client.publish(f"scenescape/data/camera/{self.cameraid}", json.dumps(self.frame_level_data))

      # 3. Format and Normalize Smart Parking Payload
      start_t = getattr(self, 'start_time', time.time())
      if 'objects' in gvametadata:
        for obj_idx, obj in enumerate(gvametadata['objects']):
          parsed_color = "silver"
          parsed_conf = 13.205
          parsed_id = 1

          # Normalizing classification dictionary
          for k, v in list(obj.items()):
            if k.startswith('classification_layer_name:') and isinstance(v, dict):
              raw_label = str(v.get('label', ''))
              if ',' in raw_label:
                try:
                  scores = [float(s.strip()) for s in raw_label.split(',')]
                  pred_idx = int(np.argmax(scores))
                  parsed_color = COLOR_LABELS[pred_idx] if pred_idx < len(COLOR_LABELS) else 'silver'
                  parsed_conf = float(scores[pred_idx])
                  parsed_id = pred_idx
                except Exception:
                  pass
              v['label'] = parsed_color
              v['confidence'] = parsed_conf
              v['label_id'] = parsed_id

          # Normalizing gva_meta tensors
          if 'gva_meta' in gvametadata and obj_idx < len(gvametadata['gva_meta']):
            gva_entry = gvametadata['gva_meta'][obj_idx]
            det_tensor = {
              "name": "detection",
              "confidence": float(obj.get('detection', {}).get('confidence', 0.927)),
              "label_id": int(obj.get('detection', {}).get('label_id', 2)),
              "label": str(obj.get('detection', {}).get('label', 'car'))
            }
            cls_tensor = {
              "name": "classification",
              "confidence": parsed_conf,
              "label_id": parsed_id,
              "label": parsed_color
            }
            gva_entry['tensor'] = [det_tensor, cls_tensor]

      gvametadata['frame_id'] = self.frame_count
      gvametadata['time'] = now_ns
      gvametadata['pipeline'] = {
        'name': 'user_defined_pipelines',
        'version': 'yolov11s',
        'instance_id': self.cameraid,
        'status': {
          'avg_fps': float(self.frame_level_data['rate']),
          'avg_pipeline_latency': None,
          'elapsed_time': round(time.time() - start_t, 2),
          'frame_fps': float(self.frame_level_data['rate']),
          'id': self.cameraid,
          'message': '',
          'start_time': start_t,
          'state': 'RUNNING'
        }
      }
      
      self.client.publish(self.sp_topic, json.dumps({'metadata': gvametadata, 'blob': ''}))
      frame.add_message(json.dumps(self.frame_level_data))
    return True
