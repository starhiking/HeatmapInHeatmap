class Basic_Config():
    def __init__(self):
        self.head_type = ''
        self.data_type = ''
        self.data_folder = 'train'
        self.test_folder = 'test'
        self.batch_size = 16
        self.num_workers = 8
        self.init_lr = 1e-4
        self.offset_lr = None
        self.num_epochs = 60
        self.decay_steps = [30, 50]
        self.val_epoch = [[30,10],[40,5],[100,1]] # 前30epoch 每10个测一次，前40epoch每5次测一次，前100epoch每1次测一次
        self.input_size = 256
        self.backbone = ''
        self.heatmap_size = 64
        self.heatmap_method = "GAUSS"
        self.heatmap_sigma = 1.5
        self.pretrained = True
        self.criterion_heatmap = 'l2'
        self.num_landmarks = 0
        self.use_gpu = True
        self.gpu_id = 0
        self.inference_indice = -1
        self.mlp_r = 2
        self.per_stack_heatmap = 1 # 每x个stack都输出一次heatmap,默认为1
    def _print(self):
        return '\n' + '\n'.join(['%s:%s' % item for item in self.__dict__.items()])

