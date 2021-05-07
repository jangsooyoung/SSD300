from datetime import datetime

import tkinter as tk
from   tkinter import *
import tkinter.font as fn
from PIL import Image, ImageTk, ImageDraw
import argparse, ntpath
import os, sys, glob
import numpy as np
import yaml
from tqdm import tqdm
import random
import xml.etree.ElementTree as ET
from box_utils import compute_target
from image_utils import random_patching, horizontal_flip
from functools import partial
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
import face_recognition.api as face_recognition

from anchor import generate_default_boxes
from box_utils import decode, compute_nms
from image_utils import ImageVisualizer
from losses import create_losses
from network import create_ssd
import click, os, glob, re, multiprocessing, sys, itertools, cv2, random, math
import xml.etree.ElementTree as xml_tree
import threading,  time, queue
import pyglet
from playsound import playsound
import contextlib
import tensorflow as tf
import gc
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
print("Num GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

cwd=os.getcwd()+ os.sep

NUM_CLASSES = 9 # model class 종류 buffer

cwd=os.getcwd()+ os.sep
class_idx_to_name = ['', 'h', 'f', 'm', 'p', 'c', 'o']
ssd = create_ssd(NUM_CLASSES, 'ssd300', 'latest', f'{cwd}/checkpoints', 'checkpoints')
class Obj:
    # xml 형식으로 저장될 내용
    def __init__(self, name, xmin, ymin, xmax, ymax, truncated=0, difficult=0, objectBox=None, objectNm=None,
                 objectNmBg=None, parent=None):
        self.name = name
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.truncated = truncated
        self.difficult = difficult
        self.objectBox = objectBox
        self.objectNm = objectNm
        self.objectNmBg = objectNmBg
        self.parent = parent

    def __str__(self):
        return "<{}<{},{} ~ {},{}>_".format(self.name, self.xmin, self.ymin, self.xmax, self.ymax)

def saveVocXml(path_file_name, width, height, object_list):
    fname = os.path.basename(path_file_name)
    xml = []
    xml.append("<annotation>")
    xml.append("  <folder>carno</folder>")
    xml.append("  <filename>{}</filename>".format(fname))
    xml.append("  <source>")
    xml.append("    <database>carno</database>")
    xml.append("    <annotation>carno</annotation>")
    xml.append("    <image>flickr</image>")
    xml.append("  </source>")
    xml.append("  <size>")
    xml.append("        <width>{}</width>".format(width))
    xml.append("        <height>{}</height>".format(height))
    xml.append("    <depth>3</depth>")
    xml.append("  </size>")
    xml.append("  <segmented>0</segmented>")

    for obj in object_list:
        if obj.parent != None:
            continue
        xml.append("  <object>")
        xml.append("    <name>{}</name>".format(obj.name))
        xml.append("    <pose>Unspecified</pose>")
        xml.append("    <truncated>{}</truncated>".format(obj.truncated))
        xml.append("    <difficult>{}</difficult>".format(obj.difficult))
        xml.append("    <bndbox>")
        xml.append("            <xmin>{}</xmin>".format(obj.xmin))
        xml.append("            <ymin>{}</ymin>".format(obj.ymin))
        xml.append("            <xmax>{}</xmax>".format(obj.xmax))
        xml.append("            <ymax>{}</ymax>".format(obj.ymax))
        xml.append("    </bndbox>")
        part_list = getPartList(obj, object_list)
        for sobj in part_list:
            xml.append("    <part>")
            xml.append("      <name>{}</name>".format(sobj.name))
            xml.append("      <bndbox>")
            xml.append("                <xmin>{}</xmin>".format(sobj.xmin))
            xml.append("                <ymin>{}</ymin>".format(sobj.ymin))
            xml.append("                <xmax>{}</xmax>".format(sobj.xmax))
            xml.append("                <ymax>{}</ymax>".format(sobj.ymax))
            xml.append("      </bndbox>")
            xml.append("    </part>")
        xml.append("  </object>")
    xml.append("</annotation>")
    f = open(path_file_name, "w")
    f.write('\n'.join(xml))
    f.close()

def makeFunc(f, x):
  return lambda: f(x)

last_save_time ='x'
def saveImage(fname,  boxes, classnames):
    object_list = []
    for ind in range(len(classnames)):
        b = boxes[ind]
        object_list.append(Obj(classnames[ind], b[0], b[1], b[2], b[3]))
    saveVocXml(fname, w, h, object_list)

#@#tf.function
def predict1(img):
    global  default_boxes
    img300 = cv2.resize(img, (300, 300))
    img300 = cv2.cvtColor(img300, cv2.COLOR_BGR2RGB)
    img300 = (img300 / 127.0) - 1.0
    #return predict([img300], default_boxes)
    return predict(np.array([img300]), default_boxes)

def predict(imgs, default_boxes):
    #print("ssd begin>" + type(imgs).__name__ )
    confs, locs = ssd(imgs)
    timelog("ssd end" )
    confs = tf.squeeze(confs, 0)
    locs = tf.squeeze(locs, 0)
    confs = tf.math.softmax(confs, axis=-1)
    classes = tf.math.argmax(confs, axis=-1)
    scores = tf.math.reduce_max(confs, axis=-1)
    boxes = decode(default_boxes, locs)

    out_boxes = []
    out_labels = []
    out_scores = []

    timelog("decode begin")
    for c in range(1, NUM_CLASSES):
        cls_scores = confs[:, c]
        score_idx = cls_scores > 0.75
        # cls_boxes = tf.boolean_mask(boxes, score_idx)
        # cls_scores = tf.boolean_mask(cls_scores, score_idx)
        cls_boxes = boxes[score_idx]
        cls_scores = cls_scores[score_idx]

        nms_idx = compute_nms(cls_boxes, cls_scores, 0.45, 200)
        cls_boxes = tf.gather(cls_boxes, nms_idx)
        cls_scores = tf.gather(cls_scores, nms_idx)
        cls_labels = [c] * cls_boxes.shape[0]

        out_boxes.append(cls_boxes)
        out_labels.extend(cls_labels)
        out_scores.append(cls_scores)
    timelog("decode  end" )

    out_boxes = tf.concat(out_boxes, axis=0)
    out_scores = tf.concat(out_scores, axis=0)

    boxes = tf.clip_by_value(out_boxes, 0.0, 1.0).numpy()
    classes = np.array(out_labels)
    scores = out_scores.numpy()
    timelog("predict  return" )
    return boxes, classes, scores

def getPartList(obj, object_list):
    part_list = []
    for o in object_list:
        if o.parent == obj:
            part_list.append(o)
    return part_list
    
def timelog(s):
    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'), s)

if __name__ == '__main__':
    cfg = yaml.load( open(f'./config.yml'))
    print(cfg)
    config = cfg['SSD300']
    default_boxes = generate_default_boxes(config)
    if len(sys.argv) <= 1:
        print('사용법  python eval.py input_jpg ')
    else:
        fname = sys.argv[1]
        img = cv2.imread(fname, cv2.IMREAD_COLOR)
        w = len(img[0])
        h = len(img)
        boxes_float, classes, scores = predict1(img)
        
        classnames = [ class_idx_to_name[c] for c in classes]
        boxes = [ [ int(b[0]*w), int(b[1]*h), int(b[2]*w), int(b[3]*h) ]  for b in boxes_float]
        
        fname = fname.replace(".jpg", ".xml")
        saveImage(fname, boxes, classnames)
