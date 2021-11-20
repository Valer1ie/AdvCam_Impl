import tensorflow as tf
import param_config as cfg
import math
import numpy as np
import os
from PIL import Image




def compute_transform_vec(min_scale, width, maxrotation):
    im_scale = np.random.uniform(low=min_scale, high=0.6)

    padding = (1 - im_scale) * width  # 中心化
    x_shift = np.random.uniform(-padding, padding)
    y_shift = np.random.uniform(-padding, padding)

    rotation_deg = np.random.uniform(-maxrotation, maxrotation)

    rotation = -math.pi / 2 * float(rotation_deg) / 90.
    rotation_matrix = np.array([
        [math.cos(rotation), -math.sin(rotation)],
        [math.sin(rotation), math.cos(rotation)]
    ])

    inv_scale = 1. / im_scale
    scaled_matrix = rotation_matrix * inv_scale  # 绕左上角旋转放缩

    x_center = float(width) / 2
    y_center = float(width) / 2

    x_center_shift, y_center_shift = np.matmul(scaled_matrix, [x_center, y_center])

    x_center_delta = x_center - x_center_shift
    y_center_delta = y_center - y_center_shift  # 不进行平移的偏移量

    a1, a2 = scaled_matrix[0]
    b1, b2 = scaled_matrix[1]
    a3 = x_center_delta - (x_shift / (im_scale * 2))
    b3 = y_center_delta - (y_shift / (im_scale * 2))

    return np.array([a1, a2, a3, b1, b2, b3, 0, 0]).astype(np.float32)


class Camouflage:
    def __init__(self, content_mask, content_img, input_img):
        content_height, content_width = content_img.shape[0], content_img.shape[1]
        tf_mask = tf.constant(content_mask)
        tf_mask_reverse = tf.constant(1 - content_mask)
        content_img_masked = tf.multiply(tf.constant(content_img), tf_mask_reverse)
        attack_area = tf.multiply(input_img, tf_mask)

        self.transformed_img = tf.add(content_img_masked, attack_area)
        self.background = tf.placeholder(tf.float32, (None, content_height, content_width, 3))
        self.img_with_background = self.get_img_with_background(tf_mask, content_width)
        self.resized_img = tf.image.resize_images(self.img_with_background, (224, 224))

    def get_img_with_background(self, tf_mask, width, min_scale=0.4, max_rotation=25):
        bg = tf.squeeze(self.background, [0])
        adv_img = tf.squeeze(self.transformed_img, [0])
        shift_vector = tf.py_func(compute_transform_vec, [min_scale, width, max_rotation], tf.float32)
        shift_vector.set_shape([8])
        out = tf.contrib.image.transform(adv_img, shift_vector, "BILINEAR")
        input_mask = tf.contrib.image.transform(tf_mask, shift_vector, "BILINEAR")
        back_ground_mask = 1 - input_mask
        input_with_back_ground = tf.add(tf.multiply(back_ground_mask, bg), tf.multiply(input_mask, out))

        color_shift = input_with_back_ground + input_with_back_ground * tf.constant(np.random.uniform(-0.3, 0.3))
        color_shift = tf.expand_dims(color_shift, 0)

        return tf.clip_by_value(color_shift, 0.0, 255.0)

    def get_random_background(self, height, width):
        files = os.listdir(cfg.current_back_ground)
        rand_n = np.random.randint(0, len(files))
        file_name = os.path.join(cfg.current_back_ground, files[rand_n])
        bg = np.array(Image.open(file_name).convert("RGB").resize((height, width)), dtype=np.float32)
        return np.expand_dims(bg, 0)
