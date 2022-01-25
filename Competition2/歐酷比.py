#!/usr/bin/env python
# coding: utf-8

# # Deep Learning Competition 2
# * Team : 歐🆒比
# * 107062208邱靖豪 107062132鍾皓崴 107061202蔣承軒 110062577景璞
# 
# 1.	Date Preprocess
# 在預處理資料時，我們發現原始資料集有相當不平衡的label數。其中「人」
# 、「椅子」、「車子」是最多的，因此我們對於其他的資料量新增了一些數量，以達成稍微平衡的功效。我們的作法為對於照片中不存在那三項的物件的話，便新增8張相同的圖，如果出現「椅子」或「車子」但沒有「人」，則新增2張一樣的圖，如果出現了「人」，則不新增任何圖片。除此之外，我們也參考了網路上對這種物件偵測資料集做preprocess的方法，分別對圖片做翻轉、隨機剪裁及縮放，還有調整對比等。除此之外，我們也有將原始的資料切出部分當成validation set，並根據validation set的結果調整我們的hyper parameter。  
# 
# 

# 2.	Model Architecture & Object detection method 
# 我們在這次的比賽中所使用的方法是YOLO v1，並且搭配上根據YOLO v2中提出的Batch Normalization以及在沒有在助教提供的程式中的技巧NMS，用來篩選匡出來的框，幫助我們過了baseline60。在feature extractor的選擇上，我們曾經使用過ResNet50, SimCLR, EfficientNet, Inception ResNet等，我們也在實驗過程中發現Inception ResNet的效果是最好的，因此便以它當成最終的feature extractor。除此之外，我們也發現到在原始的YOLO架構中，接在feature extractor後的”YOLO-specific design”，不需要像原始方法一樣接那麼多層。是由於我們的data量太少，如果還加了那麼多層的話容易有overfit的問題。我們也嘗試了EfficientNet作為feature extractor做training，分別將model freeze和finetune，在training時兩者皆有收斂且testing set上的效果看起來也不錯，但是最終的成績不太好，因此就放棄了這個model嘗試。
# 實際上我們有嘗試過在pretrain的feature extractor後的網路組合有: (1)4-layer 2D-convolution + 2-layer MLP (2) 2-layer 2D-convolution + 2-layer MLP (3) 2-layer 2D-convolution + 3-layer MLP (4) 2-layer 2D-convolution + 4-layer MLP (5) max-pooling + 3-layer MLP (6) directly 3-layer MLP。在(1)的架構下，我們發現由於需要訓練多層convloution學習由pretrain feature extractor抽取的feature map，實際上學習的效率並不會特別好。基於這樣的想法，我們採用了(2)的設置，大幅降低了原先設置收斂後的loss。更進一步的，我們加深了MLP的深度，希望能學習到更多對於利用特徵分類的資訊，並得到小幅度提升。我們也發現，在(4)的架構下，多加的那一層perceptron對於會使得模型過度複雜，效果並不彰。基於某種程度簡化特徵反而會提升效能的想法，我們也嘗試了將convolution換成max pooling，希望能特別學到特徵中相對明顯的內容。這個設定也稍微提升了表現，獲得了目前我們最高的分數。最後我們也抱著嘗試的心態測試了直接接mlp的表現，但效果並不好。

# In[83]:


import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import cv2
import albumentations as A

import tensorflow_addons as tfa


# In[4]:


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Select GPU number 1
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# In[1]:


classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", 
                 "bus", "car", "cat", "chair", "cow", "diningtable", 
                 "dog", "horse", "motorbike", "person", "pottedplant", 
                 "sheep", "sofa", "train","tvmonitor"]


# In[5]:


training_data_file = open("./pascal_voc_training_data.txt", "r")
for i, line in enumerate(training_data_file):
    if i >5:
        break
    line = line.strip()
    print(line)


# In[84]:


# common params
IMAGE_SIZE = 299
BATCH_SIZE = 32
NUM_CLASSES = 20
MAX_OBJECTS_PER_IMAGE = 20

# dataset params
DATA_PATH = './pascal_voc_training_data.txt'
IMAGE_DIR = './VOCdevkit_train/VOC2007/JPEGImages/'

# model params
CELL_SIZE = 7
BOXES_PER_CELL = 2
OBJECT_SCALE = 1
NOOBJECT_SCALE = 0.5
CLASS_SCALE = 1
COORD_SCALE = 5

# training params
LEARNING_RATE = 1e-5
EPOCHS = 30


# # Data augmentation
# * Randomly flip/resize/adjust hue and saturation.

# In[7]:


transform = A.Compose([
    A.RandomSizedBBoxSafeCrop(IMAGE_SIZE, IMAGE_SIZE, erosion_rate=0.2, p=0.5),
    A.Flip(p=0.5),
    A.HueSaturationValue(p=1.0),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1.0)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))


# # Balance data, split data, get label

# In[55]:


class DatasetGenerator:
    """
    Load pascalVOC 2007 dataset and creates an input pipeline.
    - Reshapes images into 448 x 448
    - converts [0 1] to [-1 1]
    - shuffles the input
    - builds batches
    """

    def __init__(self):
        self.pipeline = []
        self.image_names= []
        self.record_list= []
        self.object_num_list = []
        # filling the record_list
        input_file = open(DATA_PATH, 'r')

        input_file_all = []
        for line in input_file:
            input_file_all.append(line)
        input_file_train, input_file_test = train_test_split(input_file_all, test_size=0.2, random_state=0)

        self.pipeline = [input_file_train, input_file_test]

        for idx, pipe in enumerate(self.pipeline):
            image_names_ = []
            record_list_ = []
            object_num_list_ = []


            for line in pipe:
                line = line.strip()
                ss = line.split(' ')

                count = 1
                flag = 0
                for i in range(5, len(ss), 5):
                    if(int(ss[i]) == 14):
                        count = 1
                        break
                    elif(int(ss[i]) == 6 or int(ss[i]) == 8):
                        count = 2
                        flag = 1
                    else:
                        if(flag):
                            count = 2
                        else:
                            count = 8

                for _ in range(count):
                    image_names_.append(ss[0])

                    record_list_.append([float(num) for num in ss[1:]])

                    object_num_list_.append(min(len(record_list_[-1])//5, 
                                                          MAX_OBJECTS_PER_IMAGE))
                    if len(record_list_[-1]) < MAX_OBJECTS_PER_IMAGE*5:
                        # if there are objects less than MAX_OBJECTS_PER_IMAGE, pad the list
                        record_list_[-1] = record_list_[-1] +                        [0., 0., 0., 0., 0.]*                        (MAX_OBJECTS_PER_IMAGE-len(record_list_[-1])//5)
                        
                    elif len(record_list_[-1]) > MAX_OBJECTS_PER_IMAGE*5:
                    # if there are objects more than MAX_OBJECTS_PER_IMAGE, crop the list
                        record_list_[-1] = record_list_[-1][:MAX_OBJECTS_PER_IMAGE*5]
            self.image_names.append(image_names_) # train, test
            self.record_list.append(record_list_) # train, test
            self.object_num_list.append(object_num_list_) # train, test
    
                
    def _data_preprocess(self, image_name, raw_labels, object_num):
        image_file = tf.io.read_file(IMAGE_DIR+image_name)
        image = tf.io.decode_jpeg(image_file, channels=3)

        raw_labels = tf.cast(tf.reshape(raw_labels, [-1, 5]), tf.float32)
        bboxes = raw_labels[:object_num, 0:4]
        category_ids = raw_labels[:object_num, 4]

        # image augmentation
        transformed = transform(image=image.numpy(), bboxes=bboxes.numpy(), category_ids=category_ids.numpy())

        image = tf.keras.applications.inception_resnet_v2.preprocess_input(transformed['image'])

        object_num = len(bboxes)

        bboxes = transformed['bboxes']
        bboxes = np.vstack([bboxes, np.zeros((MAX_OBJECTS_PER_IMAGE - object_num, 4))])

        xmin = bboxes[:, 0]
        ymin = bboxes[:, 1]
        xmax = bboxes[:, 2]
        ymax = bboxes[:, 3]

        class_num = transformed['category_ids']
        class_num = np.hstack([class_num, np.zeros((MAX_OBJECTS_PER_IMAGE - object_num),)])

        xcenter = (xmin + xmax) * 1.0 / 2.0
        ycenter = (ymin + ymax) * 1.0 / 2.0

        box_w = xmax - xmin
        box_h = ymax - ymin

        labels = tf.stack([xcenter, ycenter, box_w, box_h, class_num], axis=1)

        return image, tf.cast(labels, tf.float32), tf.cast(object_num, tf.int32)

    def _data_preprocess_valid(self, image_name, raw_labels, object_num):
        image_file = tf.io.read_file(IMAGE_DIR+image_name)
        image = tf.io.decode_jpeg(image_file, channels=3)

        h = tf.shape(image)[0]
        w = tf.shape(image)[1]

        width_ratio  = IMAGE_SIZE * 1.0 / tf.cast(w, tf.float32) 
        height_ratio = IMAGE_SIZE * 1.0 / tf.cast(h, tf.float32) 

        image = tf.image.resize(image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = tf.keras.applications.inception_resnet_v2.preprocess_input(image)

        raw_labels = tf.cast(tf.reshape(raw_labels, [-1, 5]), tf.float32)

        xmin = raw_labels[:, 0]
        ymin = raw_labels[:, 1]
        xmax = raw_labels[:, 2]
        ymax = raw_labels[:, 3]
        class_num = raw_labels[:, 4]

        xcenter = (xmin + xmax) * 1.0 / 2.0 * width_ratio
        ycenter = (ymin + ymax) * 1.0 / 2.0 * height_ratio

        box_w = (xmax - xmin) * width_ratio
        box_h = (ymax - ymin) * height_ratio

        labels = tf.stack([xcenter, ycenter, box_w, box_h, class_num], axis=1)

        return image, labels, tf.cast(object_num, tf.int32)

    def tf_data_preprocess(self, image_name, raw_labels, object_num):
        tf_tensor = tf.py_function(self._data_preprocess, (image_name, raw_labels, object_num), Tout=(tf.float32, tf.float32, tf.int32))
        return tf_tensor

    def generate(self):
        out = []
        flag = True
        for image_names, record_list, object_num_list in zip(self.image_names, self.record_list, self.object_num_list):
            dataset = tf.data.Dataset.from_tensor_slices((image_names, 
                                                          np.array(record_list), 
                                                          np.array(object_num_list)))
            dataset = dataset.shuffle(100000)
            if flag:
                dataset = dataset.map(self.tf_data_preprocess, num_parallel_calls = tf.data.experimental.AUTOTUNE)
                flag = False
            else:
                dataset = dataset.map(self._data_preprocess_valid, num_parallel_calls = tf.data.experimental.AUTOTUNE)
                
            dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
            dataset = dataset.prefetch(buffer_size=200)
            out.append(dataset)

        return out[0], out[1]


# # Model
# * Use inception-resnet-v2 pre-trained on imagenet as feature extractor, and make the yolo part lighter avoid overfit

# In[56]:


def conv_leaky_relu(inputs, filters, size, stride):
    x = layers.Conv2D(filters, size, stride, padding="same",
                      kernel_initializer=tf.keras.initializers.TruncatedNormal())(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    #x = tfa.activations.mish(x)
    return x


# we use inception-resnet-v2 model pre-trained on imagenet as the feature extractor.

# In[57]:


inception_resnet_v2 = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
inception_resnet_v2.trainable = False


# In[58]:


img_inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = inception_resnet_v2(img_inputs)

x = layers.MaxPooling2D(pool_size=(2, 2),strides=(2, 2), padding='valid')(x)
x = layers.Flatten()(x)
x = layers.Dense(4096, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))(x)
x = tfa.activations.mish(x)
x = layers.Dense(2048, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))(x)
x = tfa.activations.mish(x)
x = layers.Dense(1024, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))(x)
x = tfa.activations.mish(x)
outputs = layers.Dense(1470, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))(x)

YOLO = keras.Model(inputs=img_inputs, outputs=outputs, name="YOLO")


# In[59]:


YOLO.summary()


# # Calculate loss
# * Also try giou compute the loss

# In[60]:


# base boxes (for loss calculation)
base_boxes = np.zeros([CELL_SIZE, CELL_SIZE, 4])

# initializtion for each cell
for y in range(CELL_SIZE):
    for x in range(CELL_SIZE):
        base_boxes[y, x, :] = [IMAGE_SIZE / CELL_SIZE * x, 
                               IMAGE_SIZE / CELL_SIZE * y, 0, 0]

base_boxes = np.resize(base_boxes, [CELL_SIZE, CELL_SIZE, 1, 4])
base_boxes = np.tile(base_boxes, [1, 1, BOXES_PER_CELL, 1])


# In[61]:


def yolo_loss(predicts, labels, objects_num):
    """
    Add Loss to all the trainable variables
    Args:
        predicts: 4-D tensor [batch_size, cell_size, cell_size, num_classes + 5 * boxes_per_cell]
        ===> (num_classes, boxes_per_cell, 4 * boxes_per_cell)
        labels  : 3-D tensor of [batch_size, max_objects, 5]
        objects_num: 1-D tensor [batch_size]
    """

    loss = 0.
    
    #you can parallel the code with tf.map_fn or tf.vectorized_map (big performance gain!)
    for i in tf.range(BATCH_SIZE):
        predict = predicts[i, :, :, :]
        label = labels[i, :, :]
        object_num = objects_num[i]

        for j in tf.range(object_num):
            results = losses_calculation(predict, label[j:j+1, :])
            loss = loss + results

    return loss / BATCH_SIZE


# In[62]:


def iou(boxes1, boxes2):
    """calculate ious
    Args:
      boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
      boxes2: 1-D tensor [4] ===> (x_center, y_center, w, h)

    Return:
      iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
      ====> iou score for each cell
    """

    #boxes1 : [4(xmin, ymin, xmax, ymax), cell_size, cell_size, boxes_per_cell]
    boxes1 = tf.stack([boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2,
                      boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2])

    #boxes1 : [cell_size, cell_size, boxes_per_cell, 4(xmin, ymin, xmax, ymax)]
    boxes1 = tf.transpose(boxes1, [1, 2, 3, 0])

    boxes2 =  tf.stack([boxes2[0] - boxes2[2] / 2, boxes2[1] - boxes2[3] / 2,
                      boxes2[0] + boxes2[2] / 2, boxes2[1] + boxes2[3] / 2])
    
    #calculate the left up point of boxes' overlap area
    lu = tf.maximum(boxes1[:, :, :, 0:2], boxes2[0:2])
    #calculate the right down point of boxes overlap area
    rd = tf.minimum(boxes1[:, :, :, 2:], boxes2[2:])

    #intersection
    intersection = rd - lu 

    #the size of the intersection area
    inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1]

    mask = tf.cast(intersection[:, :, :, 0] > 0, tf.float32) * tf.cast(intersection[:, :, :, 1] > 0, tf.float32)

    #if intersection is negative, then the boxes don't overlap
    inter_square = mask * inter_square

    #calculate the boxs1 square and boxs2 square
    square1 = (boxes1[:, :, :, 2] - boxes1[:, :, :, 0]) * (boxes1[:, :, :, 3] - boxes1[:, :, :, 1])
    square2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])

    return inter_square/(square1 + square2 - inter_square + 1e-6)

def giou(boxes1, boxes2):
    """calculate ious
    Args:
      boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
      boxes2: 1-D tensor [4] ===> (x_center, y_center, w, h)

    Return:
      iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
      ====> iou score for each cell
    """

    #boxes1 : [4(xmin, ymin, xmax, ymax), cell_size, cell_size, boxes_per_cell]
    boxes1 = tf.stack([boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2,
                      boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2])

    #boxes1 : [cell_size, cell_size, boxes_per_cell, 4(xmin, ymin, xmax, ymax)]
    boxes1 = tf.transpose(boxes1, [1, 2, 3, 0])

    boxes2 =  tf.stack([boxes2[0] - boxes2[2] / 2, boxes2[1] - boxes2[3] / 2,
                      boxes2[0] + boxes2[2] / 2, boxes2[1] + boxes2[3] / 2])
    
    return tfa.losses.giou_loss(boxes1 ,boxes2)
def losses_calculation(predict, label):
    """
    calculate loss
    Args:
      predict: 3-D tensor [cell_size, cell_size, num_classes + 5 * boxes_per_cell]
      label : [1, 5]  (x_center, y_center, w, h, class)
    """
    label = tf.reshape(label, [-1])

    #Step A. calculate objects tensor [CELL_SIZE, CELL_SIZE]
    #turn pixel position into cell position (corner)
    min_x = (label[0] - label[2] / 2) / (IMAGE_SIZE / CELL_SIZE)
    max_x = (label[0] + label[2] / 2) / (IMAGE_SIZE / CELL_SIZE)

    min_y = (label[1] - label[3] / 2) / (IMAGE_SIZE / CELL_SIZE)
    max_y = (label[1] + label[3] / 2) / (IMAGE_SIZE / CELL_SIZE)

    min_x = tf.floor(min_x)
    min_y = tf.floor(min_y)

    max_x = tf.minimum(tf.math.ceil(max_x), CELL_SIZE)
    max_y = tf.minimum(tf.math.ceil(max_y), CELL_SIZE)
    
    #calculate mask of object with cells
    onset = tf.cast(tf.stack([max_y - min_y, max_x - min_x]), dtype=tf.int32)
    object_mask = tf.ones(onset, tf.float32)

    offset = tf.cast(tf.stack([min_y, CELL_SIZE - max_y, min_x, CELL_SIZE - max_x]), tf.int32)
    offset = tf.reshape(offset, (2, 2))
    object_mask = tf.pad(object_mask, offset, "CONSTANT")

    #Step B. calculate the coordination of object center and the corresponding mask
    #turn pixel position into cell position (center)
    center_x = label[0] / (IMAGE_SIZE / CELL_SIZE)
    center_x = tf.floor(center_x)

    center_y = label[1] / (IMAGE_SIZE / CELL_SIZE)
    center_y = tf.floor(center_y)

    response = tf.ones([1, 1], tf.float32)

    #calculate the coordination of object center with cells
    objects_center_coord = tf.cast(tf.stack([center_y, CELL_SIZE - center_y - 1, 
                             center_x, CELL_SIZE - center_x - 1]), 
                             tf.int32)
    objects_center_coord = tf.reshape(objects_center_coord, (2, 2))

    #make mask
    response = tf.pad(response, objects_center_coord, "CONSTANT")

    #Step C. calculate iou_predict_truth [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    predict_boxes = predict[:, :, NUM_CLASSES + BOXES_PER_CELL:]

    predict_boxes = tf.reshape(predict_boxes, [CELL_SIZE, 
                                               CELL_SIZE, 
                                               BOXES_PER_CELL, 4])
    #cell position to pixel position
    predict_boxes = predict_boxes * [IMAGE_SIZE / CELL_SIZE, 
                                     IMAGE_SIZE / CELL_SIZE, 
                                     IMAGE_SIZE, IMAGE_SIZE]

    #if there's no predict_box in that cell, then the base_boxes will be calcuated with label and got iou equals 0
    predict_boxes = base_boxes + predict_boxes

    iou_predict_truth = iou(predict_boxes, label[0:4])

    #calculate C tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    C = iou_predict_truth * tf.reshape(response, [CELL_SIZE, CELL_SIZE, 1])

    #calculate I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    I = iou_predict_truth * tf.reshape(response, [CELL_SIZE, CELL_SIZE, 1])

    max_I = tf.reduce_max(I, 2, keepdims=True)

    #replace large iou scores with response (object center) value
    I = tf.cast((I >= max_I), tf.float32) * tf.reshape(response, (CELL_SIZE, CELL_SIZE, 1))

    #calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    no_I = tf.ones_like(I, dtype=tf.float32) - I

    p_C = predict[:, :, NUM_CLASSES:NUM_CLASSES + BOXES_PER_CELL]

    #calculate truth x, y, sqrt_w, sqrt_h 0-D
    x = label[0]
    y = label[1]

    sqrt_w = tf.sqrt(tf.abs(label[2]))
    sqrt_h = tf.sqrt(tf.abs(label[3]))

    #calculate predict p_x, p_y, p_sqrt_w, p_sqrt_h 3-D [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    p_x = predict_boxes[:, :, :, 0]
    p_y = predict_boxes[:, :, :, 1]

    p_sqrt_w = tf.sqrt(tf.minimum(IMAGE_SIZE * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 2])))
    p_sqrt_h = tf.sqrt(tf.minimum(IMAGE_SIZE * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 3])))

    #calculate ground truth p 1-D tensor [NUM_CLASSES]
    P = tf.one_hot(tf.cast(label[4], tf.int32), NUM_CLASSES, dtype=tf.float32)

    #calculate predicted p_P 3-D tensor [CELL_SIZE, CELL_SIZE, NUM_CLASSES]
    p_P = predict[:, :, 0:NUM_CLASSES]

    #class_loss
    class_loss = tf.nn.l2_loss(tf.reshape(object_mask, (CELL_SIZE, CELL_SIZE, 1)) * (p_P - P)) * CLASS_SCALE

    #object_loss
    object_loss = tf.nn.l2_loss(I * (p_C - C)) * OBJECT_SCALE

    #noobject_loss
    noobject_loss = tf.nn.l2_loss(no_I * (p_C)) * NOOBJECT_SCALE

    #coord_loss
    coord_loss = (tf.nn.l2_loss(I * (p_x - x)/(IMAGE_SIZE/CELL_SIZE)) +
                  tf.nn.l2_loss(I * (p_y - y)/(IMAGE_SIZE/CELL_SIZE)) +
                  tf.nn.l2_loss(I * (p_sqrt_w - sqrt_w))/IMAGE_SIZE +
                  tf.nn.l2_loss(I * (p_sqrt_h - sqrt_h))/IMAGE_SIZE) * COORD_SCALE

    return class_loss + object_loss + noobject_loss + coord_loss


# # Process output
# * Select bounding boxes with confidence > 0.05

# In[63]:


def process_outputs(outputs):
    """
    Process YOLO outputs into bou
    """

    class_end = CELL_SIZE * CELL_SIZE * NUM_CLASSES
    conf_end = class_end + CELL_SIZE * CELL_SIZE * BOXES_PER_CELL
    class_probs = np.reshape(outputs[:, 0:class_end], (-1, 7, 7, 20))
    confs = np.reshape(outputs[:, class_end:conf_end], (-1, 7, 7, 2))
    boxes = np.reshape(outputs[:, conf_end:], (-1, 7, 7, 2*4))
    predicts = np.concatenate([class_probs, confs, boxes], 3)

    p_classes = predicts[0, :, :, 0:20]
    C = predicts[0, :, :, 20:22]
    coordinate = predicts[0, :, :, 22:]

    p_classes = np.reshape(p_classes, (CELL_SIZE, CELL_SIZE, 1, 20))
    C = np.reshape(C, (CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 1))
    coordinate = np.reshape(coordinate, 
                            (CELL_SIZE, 
                             CELL_SIZE,
                             BOXES_PER_CELL, 
                             4))

    P = C * p_classes
    #P's shape [7, 7, 2, 20]

    #choose the most confidence one
    max_conf = np.max(P)
    
    index = np.argmax(P)

    index = np.unravel_index(index, P.shape)

    class_num = index[3]

    max_coordinate = coordinate[index[0], index[1], index[2], :]

    xcenter = max_coordinate[0]
    ycenter = max_coordinate[1]
    w = max_coordinate[2]
    h = max_coordinate[3]

    xcenter = (index[1] + xcenter) * (IMAGE_SIZE/float(CELL_SIZE))
    ycenter = (index[0] + ycenter) * (IMAGE_SIZE/float(CELL_SIZE))

    w = w * IMAGE_SIZE
    h = h * IMAGE_SIZE

    xmin = xcenter - w/2.0
    ymin = ycenter - h/2.0

    xmax = xmin + w
    ymax = ymin + h

    confs = []
    class_nums = []
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []

    xmins.append(xmin)
    ymins.append(ymin)
    xmaxs.append(xmax)
    ymaxs.append(ymax)
    class_nums.append(class_num)
    confs.append(max_conf)

    # choose some others bounding boxes
    index_10 = np.argsort(-P.flatten())[1:20]
    print(P.flatten()[np.argsort(-P.flatten())[:5]])
    for i in index_10:
        conf = P.flatten()[i]
        if conf < 0.05:
            break
        index = np.unravel_index(i, P.shape)

        class_num = index[3]

        max_coordinate = coordinate[index[0], index[1], index[2], :]

        xcenter = max_coordinate[0]
        ycenter = max_coordinate[1]
        w = max_coordinate[2]
        h = max_coordinate[3]

        xcenter = (index[1] + xcenter) * (IMAGE_SIZE/float(CELL_SIZE))
        ycenter = (index[0] + ycenter) * (IMAGE_SIZE/float(CELL_SIZE))

        w = w * IMAGE_SIZE
        h = h * IMAGE_SIZE

        xmin = xcenter - w/2.0
        ymin = ycenter - h/2.0

        xmax = xmin + w
        ymax = ymin + h

        xmins.append(xmin)
        ymins.append(ymin)
        xmaxs.append(xmax)
        ymaxs.append(ymax)
        class_nums.append(class_num)
        confs.append(conf)

    return xmins, ymins, xmaxs, ymaxs, class_nums, confs


# * Also try nml to improve our selection

# In[88]:


def single_bbox_iou(boxes1, boxes2):
    
#     print("box1", boxes1)
#     print("box2", boxes2)
    #calculate the left up point of boxes' overlap area
    lu = tf.maximum(boxes1[0:2], boxes2[0:2])
    #calculate the right down point of boxes overlap area
    rd = tf.minimum(boxes1[2:], boxes2[2:])
    
    #print(lu)
    #print(rd)
    #intersection
    intersection = rd - lu 
    #print("intersection",intersection)
    #the size of the intersection area
    inter_square = tf.cast(intersection[0], tf.float32) * tf.cast(intersection[1], tf.float32)
    #print("inter_square", inter_square)
    mask = tf.cast(intersection[0] > 0, tf.float32) * tf.cast(intersection[1] > 0, tf.float32)

    #if intersection is negative, then the boxes don't overlap
    inter_square = mask * inter_square

    #calculate the boxs1 square and boxs2 square
    square1 = tf.abs(tf.cast((boxes1[2] - boxes1[0]) * (boxes1[3] - boxes1[1]), tf.float32))
    square2 = tf.abs(tf.cast((boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1]), tf.float32))
    #print("square1", square1)
    #print("square2", square2)
    return inter_square/(square1 + square2 - inter_square + 1e-6)
def process_outputs_nml(outputs):
    """
    Process YOLO outputs into bou
    """

    class_end = CELL_SIZE * CELL_SIZE * NUM_CLASSES
    conf_end = class_end + CELL_SIZE * CELL_SIZE * BOXES_PER_CELL
    class_probs = np.reshape(outputs[:, 0:class_end], (-1, 7, 7, 20))
    confs = np.reshape(outputs[:, class_end:conf_end], (-1, 7, 7, 2))
    boxes = np.reshape(outputs[:, conf_end:], (-1, 7, 7, 2*4))
    predicts = np.concatenate([class_probs, confs, boxes], 3)

    p_classes = predicts[0, :, :, 0:20]
    C = predicts[0, :, :, 20:22]
    coordinate = predicts[0, :, :, 22:]

    p_classes = np.reshape(p_classes, (CELL_SIZE, CELL_SIZE, 1, 20))
    C = np.reshape(C, (CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 1))
    coordinate = np.reshape(coordinate, 
                            (CELL_SIZE, 
                             CELL_SIZE,
                             BOXES_PER_CELL, 
                             4))

    P = C * p_classes
    #P's shape [7, 7, 2, 20]
    
    #choose the most confidence one
    max_conf = np.max(P)
    
    index = np.argmax(P)

    index = np.unravel_index(index, P.shape)

    class_num = index[3]

    max_coordinate = coordinate[index[0], index[1], index[2], :]

    xcenter = max_coordinate[0]
    ycenter = max_coordinate[1]
    w = max_coordinate[2]
    h = max_coordinate[3]

    xcenter = (index[1] + xcenter) * (IMAGE_SIZE/float(CELL_SIZE))
    ycenter = (index[0] + ycenter) * (IMAGE_SIZE/float(CELL_SIZE))

    w = w * IMAGE_SIZE
    h = h * IMAGE_SIZE

    xmin = xcenter - w/2.0
    ymin = ycenter - h/2.0

    xmax = xmin + w
    ymax = ymin + h

    confs = []
    class_nums = []
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []

    xmins.append(xmin)
    ymins.append(ymin)
    xmaxs.append(xmax)
    ymaxs.append(ymax)
    class_nums.append(class_num)
    confs.append(max_conf)
    
    
    index_10 = np.argsort(-P.flatten())[1:20]
    print(P.flatten()[np.argsort(-P.flatten())[:5]])
    for i in index_10:
        conf = P.flatten()[i]
        if conf < 0.05:
            break
        index = np.unravel_index(i, P.shape)

        class_num = index[3]

        max_coordinate = coordinate[index[0], index[1], index[2], :]

        xcenter = max_coordinate[0]
        ycenter = max_coordinate[1]
        w = max_coordinate[2]
        h = max_coordinate[3]

        xcenter = (index[1] + xcenter) * (IMAGE_SIZE/float(CELL_SIZE))
        ycenter = (index[0] + ycenter) * (IMAGE_SIZE/float(CELL_SIZE))

        w = w * IMAGE_SIZE
        h = h * IMAGE_SIZE

        xmin = xcenter - w/2.0
        ymin = ycenter - h/2.0

        xmax = xmin + w
        ymax = ymin + h
        
        flag = True
        for pre_boxID in range(len(xmins)):
            box_pre = tf.stack([xmins[pre_boxID], ymins[pre_boxID], xmaxs[pre_boxID], ymaxs[pre_boxID]])
            box_cur = tf.stack([xmin, ymin, xmax, ymax])
            iou_pre = single_bbox_iou(box_pre, box_cur)
            print(iou_pre)
            if(iou_pre > 0.3):
                flag = False
                break
        if(flag):
            xmins.append(xmin)
            ymins.append(ymin)
            xmaxs.append(xmax)
            ymaxs.append(ymax)
            class_nums.append(class_num)
            confs.append(conf)
        
    return xmins, ymins, xmaxs, ymaxs, class_nums, confs


# In[98]:


def visualize(img_num):
    # 1, 2, 10, 18, 22, 25, 27, 28, 29, 31, 37, 313, 316
    np_img = cv2.imread(f'./VOCdevkit_test/VOC2007/JPEGImages/{img_num:06}.jpg')
    resized_img = cv2.resize(np_img, (IMAGE_SIZE, IMAGE_SIZE))
    np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    resized_img = np_img
    np_img = np_img.astype(np.float32)
    # np_img = np_img / 255.0 * 2 - 1

    np_img = tf.keras.applications.inception_resnet_v2.preprocess_input(np_img)

    np_img = np.reshape(np_img, (1, IMAGE_SIZE, IMAGE_SIZE, 3))

    y_pred = YOLO(np_img, training=False)
    xmins, ymins, xmaxs, ymaxs, class_nums, confs = process_outputs_nml(y_pred)

    for i in range(len(xmins)):
        xmin, ymin, xmax, ymax, class_num, conf = xmins[i], ymins[i], xmaxs[i], ymaxs[i], class_nums[i], confs[i]
        class_name = classes_name[class_num]
        color = (255, 0, 255)
        if i == 0:
            color = (0, 255, 255)
        elif i==1:
            color = (0, 0, 255)
        elif i==2:
            color = (0, 255, 0)
        elif i==3:
            color = (255, 0, 0)
        else:
            color = (255, 255, 0)
        cv2.rectangle(resized_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        cv2.putText(resized_img, class_name, (0, 200-30*i), 2, 1, color, 2)

    plt.imshow(resized_img)
    plt.show()


# # Create dataset, Reload checkpoint

# In[100]:


dataset_train, dataset_val = DatasetGenerator().generate()


# In[73]:


optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
train_loss_metric = tf.keras.metrics.Mean(name='loss')


# In[74]:


ckpt = tf.train.Checkpoint(epoch=tf.Variable(0), net=YOLO)
manager = tf.train.CheckpointManager(ckpt, './ckpts/fuck', max_to_keep=50, checkpoint_name='yolo')


# In[75]:


# ckpt.restore(manager.latest_checkpoint)
#ckpt.restore('./ckpts/YOLO\\yolo-44')


# # Training
# * Converge before 50 epochs

# In[76]:


@tf.function
def train_step(image, labels, objects_num):
    with tf.GradientTape() as tape:
        outputs = YOLO(image)
        class_end = CELL_SIZE * CELL_SIZE * NUM_CLASSES
        conf_end = class_end + CELL_SIZE * CELL_SIZE * BOXES_PER_CELL
        class_probs = tf.reshape(outputs[:, 0:class_end], (-1, 7, 7, 20))
        confs = tf.reshape(outputs[:, class_end:conf_end], (-1, 7, 7, 2))
        boxes = tf.reshape(outputs[:, conf_end:], (-1, 7, 7, 2*4))
        predicts = tf.concat([class_probs, confs, boxes], 3)

        loss = yolo_loss(predicts, labels, objects_num)
        train_loss_metric(loss)

    grads = tape.gradient(loss, YOLO.trainable_weights)
    optimizer.apply_gradients(zip(grads, YOLO.trainable_weights))


# In[77]:


@tf.function
def valid_step(image, labels, objects_num):
    outputs = YOLO(image, training=False)
    class_end = CELL_SIZE * CELL_SIZE * NUM_CLASSES
    conf_end = class_end + CELL_SIZE * CELL_SIZE * BOXES_PER_CELL
    class_probs = tf.reshape(outputs[:, 0:class_end], (-1, 7, 7, 20))
    confs = tf.reshape(outputs[:, class_end:conf_end], (-1, 7, 7, 2))
    boxes = tf.reshape(outputs[:, conf_end:], (-1, 7, 7, 2*4))
    predicts = tf.concat([class_probs, confs, boxes], 3)

    return yolo_loss(predicts, labels, objects_num)


# In[85]:


print("{}, start training.".format(datetime.now()))
for i in range(EPOCHS):
    train_loss_metric.reset_states()
    ckpt.epoch.assign_add(1)
    
    for idx, (image, labels, objects_num) in enumerate(dataset_train):
        print('\r Batch: {}'.format(idx), end='')
        train_step(image, labels, objects_num)

    print()
    print("{}, Epoch {}: loss {:.2f}".format(datetime.now(), int(ckpt.epoch), train_loss_metric.result()))

    save_path = manager.save()
    print("Saved checkpoint for epoch {}: {}".format(int(ckpt.epoch), save_path))

    print("start validation...")
    valid_loss = 0
    for idx, (image, labels, objects_num) in enumerate(dataset_val):
        print('\r Batch: {}'.format(idx), end='')
        valid_loss += valid_step(image, labels, objects_num)
    print()
    print("Epoch {}: validation loss {:.2f}".format(int(ckpt.epoch), (valid_loss/123)))
    
    # if int(ckpt.epoch) % 5 == 0:
    visualize(2)
    visualize(1)


# In[ ]:


# 15 seems good


# In[25]:


ckpt = tf.train.Checkpoint(net=YOLO)
ckpt.restore('./ckpts/YOLO/yolo-13')


# In[101]:


for i in [1, 2, 10, 18, 22, 25, 27, 28, 29, 31, 37, 313, 316]:
    visualize(i)


# # Generate output

# In[28]:


test_img_files = open('./comp2/pascal_voc_testing_data.txt')
test_img_dir = './comp2/VOCdevkit_test/VOC2007/JPEGImages/'
test_images = []

for line in test_img_files:
    line = line.strip()
    ss = line.split(' ')
    test_images.append(ss[0])

test_dataset = tf.data.Dataset.from_tensor_slices(test_images)

def load_img_data(image_name):
    image_file = tf.io.read_file(test_img_dir+image_name)
    image = tf.image.decode_jpeg(image_file, channels=3)

    h = tf.shape(image)[0]
    w = tf.shape(image)[1]

    image = tf.image.resize(image, size=[IMAGE_SIZE, IMAGE_SIZE])
    image = tf.keras.applications.inception_resnet_v2.preprocess_input(image)

    return image_name, image, h, w

test_dataset = test_dataset.map(load_img_data, num_parallel_calls = tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(32)


# In[29]:


@tf.function
def prediction_step(img):
    return YOLO(img, training=False)


# In[32]:


output_file = open('./test_predictionf2.txt', 'w')

for img_name, test_img, img_h, img_w in test_dataset:
    batch_num = img_name.shape[0]
    for i in range(batch_num):
        xmins, ymins, xmaxs, ymaxs, class_nums, confs = process_outputs(prediction_step(test_img[i:i+1]))

        for j in range(len(xmins)):
            xmin, ymin, xmax, ymax, class_num, conf = xmins[j], ymins[j], xmaxs[j], ymaxs[j], class_nums[j], confs[j]
            xmin, ymin, xmax, ymax = xmin*(img_w[i:i+1]/IMAGE_SIZE), ymin*(img_h[i:i+1]/IMAGE_SIZE), xmax*(img_w[i:i+1]/IMAGE_SIZE), ymax*(img_h[i:i+1]/IMAGE_SIZE)

            #img filename, xmin, ymin, xmax, ymax, class, confidence
            output_file.write(img_name[i:i+1].numpy()[0].decode('ascii')+" %d %d %d %d %d %f\n" %(xmin, ymin, xmax, ymax, class_num, conf))

output_file.close()


# In[33]:


import sys
sys.path.insert(0, './evaluate')


# In[34]:


#import evaluate
#evaluate.evaluate("input prediction file name", "desire output csv file name")
execfile("./comp2/evaluate/evaluate.py")
evaluate('./test_predictionf2.txt', './output_filef2.csv')


# In[ ]:




