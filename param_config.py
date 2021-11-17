import os

# configuration
current_img_path = ''
current_style_seg_path = ''
current_result_dir = ''
current_style_image_path = ''
current_seg_path = ''
current_back_ground = ''


class Config:
    def __init__(self, args):
        attack_data_dir = 'physical-attack-data'
        attack_res_dir = 'attack_result'
        root_dir = os.getcwd()
        self.content_dir = os.path.join(root_dir, attack_data_dir, 'content', args.content)
        self.style_seg_path = os.path.join(root_dir, attack_data_dir, 'style-mask')
        self.style_image_path = os.path.join(root_dir, attack_data_dir, 'style', args.style_content)
        self.result_dir = os.path.join(root_dir, attack_res_dir, args.result_dir)
        self.content_seg_path = os.path.join(root_dir, attack_data_dir, 'content-mask')
        self.bg_path = os.path.join(root_dir,attack_data_dir, args.content + '_bg')

    def get_contents(self):
        contents = os.listdir(self.content_dir)
        return [os.path.join(self.content_dir, x) for x in contents]

    def set_paths(self, args, content_name):
        content_not_jpg = content_name.split('.')[0]
        current_img_path = os.path.join(self.content_dir, content_name)
        current_seg_path = os.path.join(self.content_seg_path, content_name)
        current_style_seg_path = os.path.join(self.style_seg_path, content_name)
        current_style_image_path = os.path.join(self.style_image_path, content_name)
        current_result_dir = os.path.join(self.result_dir, content_name,
                                          '_' + str(args.attack_weight) + '_' + content_not_jpg)
        if not os.path.exists(current_result_dir):
            os.mkdir(current_result_dir)
