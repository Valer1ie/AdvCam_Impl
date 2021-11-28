# AdvCam_Impl

本次复现源码[参照](https://github.com/RjDuan/AdvCam-Hide-Adv-with-Natural-Styles)

[视频讲解及ppt](https://pan.baidu.com/s/1t1oPbQMWha0zr6GoB7neZQ)

提取码 5467

## 环境要求：

**平台**：windows10

系统环境：

- `CUDA==9.0.176 `
- `cudnn==7.0.5`

python环境：

- `python3.6`
- `tensorflow-gpu==1.8.1`
- `numpy==1.14.1`
- `Pillow==6.2.1`
- `argparse==1.1`

卷积神经网络

- [vgg19](https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg19.py) 目前支持tensorflow v1

- 模型文件[VGG.npy](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs)

目录

|--physical-attack-data<br/>|	|--background <br/>|	|--content<br/>|	|--content-mask<br/>|	|--style<br/>|	|--style-mask<br/>|--CNN<br/>|	|--vgg19.py<br/>|	|--VGG.npy<br/>|--attack_result

## 输入说明

项目使用argparse库作输入，参数类型以及输入含义包含在AdvCam_main.py中 。

**注意**：

- 在配置好所需环境和保持项目文件结构的情况下不需要额外输入，项目默认参数为对stop-sign图片进行对抗生成。
- 可以使用--p参数设置cpu或gpu运行，其中--p=0为cpu，--p=1为gpu，项目默认gpu运行

## 代码解释

#### **总体思路**：

项目的关键在于构建损失函数的计算。论文中损失函数由Style loss, Content loss, Smoothness loss 以及 Attack loss 组成。下面将分别介绍损失函数的计算方式以及实现。

##### 损失函数

**Style loss**

图像的Style对应的卷积层为conv1_1, conv2_1, conv3_1, conv4_1, conv5_1, 处理方式为分别对原图像和攻击后生成的图像将5个卷积层的特征向量，在经过遮罩处理之后转化为格拉姆矩阵，以此计算距离。难点（其实也不太难）在于遮罩图像是不会变化的，我们需要将图像高度、宽度和vgg19中卷积层提取之后的池化操作的维度进行匹配，同时，也需要近似匹配卷积层的卷积操作，以对每个卷积层的特征向量进行mask处理。

```python
vgg_style.fprop(tf.constant(style_img), include_top=False)
            style_layers_conv = [vgg_style.conv1_1, vgg_style.conv2_1, vgg_style.conv3_1, vgg_style.conv4_1, vgg_style.conv5_1] # 其中vgg_style为vgg19的实例化对象
            style_img_style_layers_conv_fv = sess.run(style_layers_conv) # 向量计算
            style_img_style_layers = [tf.constant(fv) for fv in style_img_style_layers_conv_fv] # 这个是对于原图像的style图像进行提取的结果，对原图像也要进行相同的操作，同时也要对攻击后的图像也进行相同的操作                                
```

所以，我们先获取所有卷积层的名称，在循环中，不断更新content_mask或者style_mask图片的大小，与对应的卷积层获得的特征向量相乘，得到遮挡后的向量组，将向量组转化为格拉姆矩阵并标准化以便于计算距离，并且，这里有第二个难点（其实主要是麻烦点）：因为我们需要对mask（攻击）和inv_mask（非攻击）的区域分开进行特征（也就是距离），在计算inv_mask的时候，可能会出现mask图像全0的情况，在格拉姆矩阵标准化时有除零风险。原理上我们默认使用style_mask进行遮罩处理，当style_mask的均值为0时，我们将原图像和原遮罩作为特征图像的替代，计算非攻击区域的差异，并且如果当前的使用的遮罩均值还是0，则不用进行标准化，进而避免了除0问题。

```python
# 当前层名称为池化层时，池化操作将特征图像的高度和宽度作天花板除以2操作，所以与当前特征向量集合的维度进行匹配时必须将mask图像进行相应的resize，这里使用双线性插值进行缩放处理
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
                
# 同样的，在遇到卷积层时，使用tensorflow的平均池化操作代替卷积操作(3*3卷积核对应ksize=[1,3,3,1])
elif 'conv' in layer_name:
    for i in range(len(content_masks)):
        content_masks[i] = tf.nn.avg_pool(tf.pad(content_masks[i], [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), strides=[1, 1, 1, 1], ksize=[1, 3, 3, 1], padding="VALID")
        style_masks[i] = tf.nn.avg_pool(tf.pad(style_masks[i], [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), strides=[1, 1, 1, 1], ksize=[1, 3, 3, 1], padding="VALID")
```



```python
# 难点二处理：这里展示的是每一个特征卷积层计算特征差的部分
for content_mask, style_mask in zip(content_masks, style_masks):
    # 首先计算 x_prime 的格拉姆矩阵
    gram_matrix_variable = gram_matrix(tf.multiply(style_layer, content_mask))
    content_mask_mean = tf.reduce_mean(content_mask)
    gram_matrix_variable = tf.cond(tf.greater(content_mask_mean, 0.),
                                   lambda: gram_matrix_variable / (                  									tf.to_float(tf.size(style_layer)) * content_mask_mean),
                                   lambda: gram_matrix_variable) # 标准化并避免除零操作

    # 因为当计算not_attack区域的差异时，style_mask会出现全0的情况，此时可以计算原图片的content_mask的格拉姆矩阵
    # 具体算法为：如果 style_mask = 0 ，且 content_mask > 0 则全当前 style_mask_mean 为 content_mask 的均值
    # 且当前特征集为原图片的特征集，格拉姆矩阵为原图片的特征图的格拉姆矩阵
    # 反之，则当前使用原图片的特征图计算格拉姆矩阵
    current_style_mask_mean = tf.reduce_mean(style_mask)
    style_mask_mean = tf.cond(
        tf.logical_and(tf.greater(content_mask_mean, 0.), tf.equal(current_style_mask_mean, 0.)),
        lambda: tf.reduce_mean(content_mask),
        lambda: tf.reduce_mean(style_mask)
    ) # 计算应当使用的均值（用于标准化）
    current__conv = tf.cond(
        tf.logical_and(tf.greater(content_mask_mean, 0.), tf.equal(current_style_mask_mean, 0.)),
        lambda: content_layer_conv,
        lambda: style_layer_conv
    ) # 得到当前应当使用的遮罩
    gram_matrix_const = tf.cond(
        tf.logical_and(tf.greater(content_mask_mean, 0.), tf.equal(current_style_mask_mean, 0.)),
        lambda: gram_matrix(tf.multiply(content_layer_conv, content_mask)),
        lambda: gram_matrix(tf.multiply(style_layer_conv, style_mask))
    )
    # 避免除0操作
    gram_matrix_const = tf.cond(tf.greater(style_mask_mean, 0.),
                                lambda: gram_matrix_const / (
                                    tf.to_float(tf.size(current__conv)) * style_mask_mean),
                                lambda: gram_matrix_const)
    diff_style_sum = tf.reduce_mean(tf.squared_difference(gram_matrix_const, gram_matrix_variable)) * content_mask_mean # 计算两矩阵的距离
    style_layer_loss += diff_style_sum
```

**为什么用gram矩阵**

因为gram矩阵可以表示featuremap之间的关系，而低层featuremap可以表示纹理信息，纹理的组合可以表示图片的风格，所以用gram矩阵的差异来表示风格的差异。

**Content loss**

vgg19中最可以代表图像Content特征的卷积层为conv4_2，

![layers](README.assets/image-20211126230551774.png)

可以直接将攻击后图像与原图像的conv4_2层处理后的特征图形进行距离计算得到Content loss。

**为什么Content loss 不用gram矩阵**

因为Content loss不在于图片的局部性特征，而是图片的概括性特征，且主要是形状特征，所以不需要各个特征之间的关联，更重要的是对应位置的概括性特征是否保持，所以需要直接计算对应featuremap的差异。

```python
 lost_content = tf.reduce_mean(
                tf.squared_difference(content_layer_const, content_layer)) * cfg.content_weight 		# cfg.content_weight为Content_loss的权重值
```

**Smooth loss**

为了避免攻击图像过于突兀，所以引入了Smooth loss， 即计算每个像素点与相邻像素点的差值平方和。

```python
sm_loss = tf.reduce_sum(
        np.square(output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) + np.square(output[:, :-1, :-1, :] - output[:, 1:, :-1, :])) / 2		#除以2取平均差值和
```

**Attack loss**

攻击损失即计算扰动以及适应性变换后的logits层与原图像的差异，按文中描述分为两种方式

- 有目标的攻击：计算原图像标签与攻击后的logits的交叉熵对损失的贡献为负，目标标签与攻击后的交叉熵对损失贡献为正
- 无目标的攻击：只包含原图像与攻击后logits的交叉熵损失负向贡献部分

```python
  targeted = cfg.targeted
    balance = 5
    orig_pred = np.eye(1000)[orig]  # eye操作升维得到对家矩阵，取index=orig得到与1*1000且index=orig为1的向量，便于计算交叉熵
    loss_1 = -1 * tf.nn.softmax_cross_entropy_with_logits_v2(labels=orig_pred, logits=pred) # 计算交叉熵，下同
    if targeted:
        target = cfg.target
        target = np.eye(1000)[target]

        loss_2 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target, logits=pred)
        loss_attack = tf.reduce_sum(balance * loss_2 + loss_1) * cfg.current_attack_weight
        return loss_attack
    else:
        return loss_1 * cfg.current_attack_weight
```

#### 现实环境适应

1. 将对抗攻击图像进行随机仿射变换
2. 将变换后的攻击图像与背景图像进行叠加
3. 进行颜色微调后颜色校正

```python
	im_scale = np.random.uniform(low=min_scale, high=0.6) # min_scale 默认0.4

    padding = (1 - im_scale) * width  # 中心化
    x_shift = np.random.uniform(-padding, padding)
    y_shift = np.random.uniform(-padding, padding)

    rotation_deg = np.random.uniform(-maxrotation, maxrotation)

    rotation = (math.pi / 2) * float(rotation_deg) / 90.
    rotation_matrix = np.array([
        [math.cos(-rotation), -math.sin(-rotation)],
        [math.sin(-rotation), math.cos(-rotation)]
    ])

    inv_scale = 1. / im_scale
    scaled_matrix = rotation_matrix * inv_scale  # 绕左上角旋转放缩

    x_center = float(width) / 2
    y_center = float(width) / 2

    x_center_shift, y_center_shift = np.matmul(scaled_matrix, np.array([x_center, y_center]), )

    x_center_delta = x_center - x_center_shift
    y_center_delta = y_center - y_center_shift  # 不进行平移的偏移量

    a1, a2 = scaled_matrix[0]
    b1, b2 = scaled_matrix[1]
    a3 = x_center_delta - (x_shift / (im_scale * 2))
    b3 = y_center_delta - (y_shift / (im_scale * 2))

    return np.array([a1, a2, a3, b1, b2, b3, 0, 0]).astype(np.float32)
```



```python

    shift_vector = tf.py_func(compute_transform_vec, [min_scale, width, max_rotation], tf.float32) # 向量计算算子生成
    shift_vector.set_shape([8]) # 变换向量
    out = tf.contrib.image.transform(adv_img, shift_vector, "BILINEAR") 
    input_mask = tf.contrib.image.transform(tf_mask, shift_vector, "BILINEAR")
    back_ground_mask = 1 - input_mask
    input_with_back_ground = tf.add(tf.multiply(back_ground_mask, bg), tf.multiply(input_mask, out)) # 叠加到背景图片上
    
    color_shift = input_with_back_ground + input_with_back_ground * tf.constant(np.random.uniform(-0.3, 0.3)) # 颜色变换
    color_shift = tf.expand_dims(color_shift, 0)
    clip_img = tf.clip_by_value(color_shift, 0.0, 255.0) # 颜色校正
```


