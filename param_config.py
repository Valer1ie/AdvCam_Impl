import os

# configuration
current_img_path = ''
current_style_seg_path = ''
current_result_dir = ''
current_style_image_path = ''
current_seg_path = ''
current_back_ground = ''
true_label = 0
sm_weight = 0
current_attack_weight = 0
content_weight = 0
style_weight = 0
targeted = True
target = 0
max_iter = 0
learning_rate = 1000
save_iter = 1


class Config:
    def __init__(self, args):
        attack_data_dir = 'physical-attack-data'
        attack_res_dir = 'attack_result'
        root_dir = os.getcwd()
        self.content_dir = os.path.join(root_dir, attack_data_dir, 'content', args.content)
        self.style_seg_path = os.path.join(root_dir, attack_data_dir, 'style-mask')
        self.style_image_path = os.path.join(root_dir, attack_data_dir, 'style', args.style_content)
        self.result_dir = os.path.join(root_dir, attack_res_dir, args.result_dir)
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)
        self.content_seg_path = os.path.join(root_dir, attack_data_dir, 'content-mask')
        global sm_weight
        global content_weight
        global style_weight
        global target
        global targeted
        global max_iter
        global learning_rate
        global save_iter
        global current_back_ground
        content_weight = args.content_weight
        style_weight = args.style_weight
        current_back_ground = os.path.join(root_dir, attack_data_dir, 'background', args.background + '_bg')
        save_iter = args.save_iter
        learning_rate = args.learning_rate
        max_iter = args.max_iter
        target = args.target
        targeted = args.targeted == 1
        sm_weight = args.sm_weight


    def get_contents(self):
        contents = os.listdir(self.content_dir)
        return [os.path.join(self.content_dir, x) for x in contents]

    def set_paths(self, args, content_name):
        global current_style_seg_path
        global current_seg_path
        global current_img_path
        global current_result_dir
        global current_style_image_path
        content_not_jpg = content_name.split('.')[0]
        current_img_path = os.path.join(self.content_dir, content_name)
        current_seg_path = os.path.join(self.content_seg_path, content_name)
        current_style_seg_path = os.path.join(self.style_seg_path, content_name)
        current_style_image_path = os.path.join(self.style_image_path, content_name)
        current_result_dir = os.path.join(self.result_dir, content_not_jpg,
                                          '_' + str(args.attack_weight) + '_' + content_not_jpg)
        if not os.path.exists(os.path.join(self.result_dir, content_not_jpg)):
            os.mkdir(os.path.join(self.result_dir, content_not_jpg))
        if not os.path.exists(current_result_dir):
            os.mkdir(current_result_dir)
