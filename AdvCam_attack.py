import tensorflow as tf
import param_config as cfg
import numpy as np
from PIL import Image
import math
from CNN.vgg import Vgg19
from functools import reduce


def load_imgs(target_size=(400, 400)):
    _content_img = np.array(Image.open(cfg.current_img_path).convert("RGB").resize(target_size), dtype=np.float32)
    content_height, content_width = _content_img.shape[0], _content_img.shape[1]
    _content_img = _content_img.reshape((1, content_height, content_width, 3)).astype(np.float32)
    _style_img = np.array(Image.open(cfg.current_style_image_path).convert("RGB"), dtype=np.float32)
    style_height, style_width = _style_img.shape[0], _style_img.shape[1]
    _style_img = _style_img.reshape((1, style_height, style_width, 3)).astype(np.float32)
    # 将黑白灰分为 黑/白
    content_mask_seg = np.array(
        Image.open(cfg.current_seg_path).convert("RGB").resize(target_size, resample=Image.BILINEAR),
        dtype=np.float32) // 245.0
    style_mask_seg = np.array(Image.open(cfg.current_style_seg_path).convert("RGB").resize((style_height, style_width),
                                                                                           resample=Image.BILINEAR),
                              dtype=np.float32) // 245.0
    # 区分采样区域 并把两张 mask 分别分成两组图 即攻击区域和非攻击区域

    process_type = ['attack', 'not_attack']
    _color_content_masks = []
    _color_style_masks = []
    masks = [_color_content_masks, _color_style_masks]
    source = [content_mask_seg, style_mask_seg]
    for p_type in process_type:
        for i in range(len(masks)):
            m_r = [[]]
            m_g = [[]]
            m_b = [[]]
            if p_type == 'attack':
                m_r = (source[i][:, :, 0] > 0.8).astype(np.uint8)
                m_g = (source[i][:, :, 1] > 0.8).astype(np.uint8)
                m_b = (source[i][:, :, 2] > 0.8).astype(np.uint8)
            elif p_type == 'not_attack':
                m_r = (source[i][:, :, 0] < 0.5).astype(np.uint8)
                m_g = (source[i][:, :, 1] < 0.5).astype(np.uint8)
                m_b = (source[i][:, :, 2] < 0.5).astype(np.uint8)
            array = np.multiply(np.multiply(m_r, m_g), m_b).astype(np.float32)
            masks[i].append(tf.expand_dims(tf.expand_dims(tf.constant(array), 0), -1))

    return _content_img, _style_img, _color_content_masks, _color_style_masks


def gram_matrix(activations):
    height = tf.shape(activations)[1]
    width = tf.shape(activations)[2]
    color = tf.shape(activations)[3]
    gram = tf.transpose(activations, [0, 3, 1, 2])
    gram = tf.reshape(gram, [color, width * height])
    gram = tf.matmul(gram, gram, transpose_b=True)
    return gram


def sync_masks(layer_name, content_masks, style_masks, content_mask_height, content_mask_width, style_mask_height,
               style_mask_width):
    if "pool" in layer_name:
        content_mask_width, content_mask_height = int(math.ceil(content_mask_width / 2)), int(
            math.ceil(content_mask_height / 2))
        style_mask_width, style_mask_height = int(math.ceil(style_mask_width / 2)), int(
            math.ceil(style_mask_height / 2))

        for i in range(len(content_masks)):
            content_masks[i] = tf.image.resize_bilinear(content_masks[i],
                                                        tf.constant((content_mask_height, content_mask_width)))
            style_masks[i] = tf.image.resize_bilinear(style_masks[i],
                                                      tf.constant((style_mask_height, style_mask_width)))

    elif 'conv' in layer_name:
        for i in range(len(content_masks)):
            content_masks[i] = tf.nn.avg_pool(tf.pad(content_masks[i], [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"),
                                              strides=[1, 1, 1, 1],
                                              ksize=[1, 3, 3, 1], padding="VALID")
            style_masks[i] = tf.nn.avg_pool(tf.pad(style_masks[i], [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"),
                                            strides=[1, 1, 1, 1],
                                            ksize=[1, 3, 3, 1], padding="VALID")
    return content_masks, style_masks, content_mask_height, content_mask_width, content_mask_height, style_mask_height, style_mask_width


def compute_style_loss(img_style_layers_conv, style_img_style_conv,
                       style_layers, content_masks, style_masks):
    with tf.name_scope('style_loss'):
        loss_styles = []
        layer_index = 0
        _, content_mask_height, content_mask_width, _ = content_masks[1].get_shape().as_List()

        _, style_mask_height, style_mask_width, _ = style_masks[0].get_shape().as_List()

        all_layer_names = [l.name for l in Vgg19().get_all_layers()]

        for layer_name in all_layer_names:
            layer_name = layer_name[layer_name.find('/') + 1:]
            with tf.name_scope('style_loss_layer'):
                content_masks, style_masks, content_mask_height, content_mask_width, style_mask_height, style_mask_width \
                    = sync_masks(layer_name, content_masks, style_masks, content_mask_height, content_mask_width,
                                 style_mask_height, style_mask_width)

            if layer_name == style_layers[layer_index].name[style_layers[layer_index].name.find('/') + 1:]:
                style_layer = style_layers[layer_index]
                content_layer_conv = img_style_layers_conv[layer_index]
                style_layer_conv = style_img_style_conv[layer_index]

                layer_index += 1

                style_layer_loss = 0.
                for content_mask, style_mask in zip(content_masks, style_masks):

                    # 首先计算 x_prime 的格拉姆矩阵
                    gram_matrix_variable = gram_matrix(tf.multiply(style_layer, content_mask))
                    content_mask_mean = tf.reduce_mean(content_mask)
                    gram_matrix_variable = tf.cond(tf.greater(content_mask_mean, 0.),
                                                   lambda: gram_matrix_variable / (
                                                           tf.to_float(tf.size(style_layer)) * content_mask_mean),
                                                   lambda: gram_matrix_variable)

                    # 因为当计算not_attack区域的差异时，style_mask会出现全0的情况，此时可以计算原图片的content_mask的格拉姆矩阵
                    # 具体算法为：如果 style_mask = 0 ，且 content_mask > 0 则全当前 style_mask_mean 为 content_mask 的均值
                    # 且当前特征集为原图片的特征集，格拉姆矩阵为原图片的特征图的格拉姆矩阵
                    # 反之，则当前使用原style图片的特征图计算格拉姆矩阵
                    current_style_mask_mean = tf.reduce_mean(style_mask)
                    style_mask_mean = tf.cond(
                        tf.logical_and(tf.greater(content_mask_mean, 0.), tf.equal(current_style_mask_mean, 0.)),
                        lambda: tf.reduce_mean(content_mask),
                        lambda: tf.reduce_mean(style_mask)
                    )
                    current__conv = tf.cond(
                        tf.logical_and(tf.greater(content_mask_mean, 0.), tf.equal(current_style_mask_mean, 0.)),
                        lambda: content_layer_conv,
                        lambda: style_layer_conv
                    )
                    gram_matrix_const = tf.cond(
                        tf.logical_and(tf.greater(content_mask_mean, 0.), tf.equal(current_style_mask_mean)),
                        lambda: gram_matrix(tf.multiply(content_layer_conv, content_mask)),
                        lambda: gram_matrix(tf.multiply(style_layer_conv, style_mask))
                    )
                    # 避免除0操作
                    gram_matrix_const = tf.cond(tf.greater(style_mask_mean, 0.),
                                                lambda: gram_matrix_const / (
                                                        tf.to_float(tf.size(current__conv)) * style_mask_mean),
                                                lambda: gram_matrix_const)
                    diff_style_sum = tf.reduce_mean(
                        tf.squared_difference(gram_matrix_const, gram_matrix_variable)) * content_mask_mean
                    style_layer_loss += diff_style_sum
                loss_styles.append(style_layer_loss * cfg.current_attack_weight)
    return loss_styles


def smooth_loss(output):
    sm_loss = tf.reduce_sum(tf.square(output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) + tf.square(output[:, :-1, :-1, :] - output[:, 1:, 1:, :]))
    return sm_loss * cfg.sm_weight


def attack():
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess_config.gpu_options.allow_growth = True

    content_img, style_img, color_content_masks, color_style_masks = load_imgs()
    tf_input_img = tf.Variable(content_img)  # variable

    with tf.Session(config=sess_config) as sess:
        with tf.name_scope("constant"):
            vgg_style = Vgg19()

            resized_content_img = tf.image.resize_images(tf.constant(content_img), (224, 224))
            vgg_style.fprop(resized_content_img)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            probability = sess.run(vgg_style.prob)  # softmax 层输出结果

            cfg.true_label = np.argmax(probability)  # 图像真正的分类

            vgg_style.fprop(tf.constant(content_img), include_top=False)
            style_layers_conv = [vgg_style.conv1_1, vgg_style.conv2_1, vgg_style.conv3_1, vgg_style.conv4_1,
                                 vgg_style.conv5_1]
            content_fv, style_layer_conv_fv = sess.run([vgg_style.conv4_2, style_layers_conv])
            content_layer_const = tf.constant(content_fv)
            img_style_layers_conv = [tf.constant(fv) for fv in style_layer_conv_fv]

            vgg_style.fprop(tf.constant(style_img), include_top=False)
            style_img_style_layers_conv_fv = sess.run(style_layers_conv)
            style_img_style_layers = [tf.constant(fv) for fv in style_img_style_layers_conv_fv]
            del vgg_style

        with tf.name_scope("variable"):
            vgg_var = Vgg19()
            vgg_var.fprop(tf_input_img, include_top=False)      # x_prime
            style_layers = [vgg_var.conv1_1, vgg_var.conv2_1, vgg_var.conv3_1, vgg_var.conv4_1, vgg_var.conv5_1]
            content_layer = vgg_var.conv4_2

            lost_content = tf.reduce_mean(
                tf.squared_difference(content_layer_const, content_layer)) * cfg.current_attack_weight

            lost_style_list = compute_style_loss(img_style_layers_conv, style_img_style_layers, style_layers,
                                                 color_content_masks, color_style_masks)

        lost = reduce(lambda x, y: x + y, lost_style_list)

        with tf.name_scope("attack"):
            vgg_attack = Vgg19()
