from experiments.Basic_config import Basic_Config
class Config(Basic_Config):
    def __init__(self):
        super(Config,self).__init__()
        self.head_type = 'hih'
        self.data_type = 'WFLW'
        self.data_folder = 'train'
        self.test_folder = 'test'
        self.batch_size = 16
        self.num_workers = 4
        self.init_lr = 5e-6
        self.num_epochs = 60
        self.decay_steps = [30, 50]
        self.val_epoch = [[40,5],[70,1]]
        self.input_size = 256
        self.backbone = 'hourglass'
        self.heatmap_size = 64
        self.num_stack = 4
        self.num_layer = 4
        self.num_feature = 256
        self.offset_size = 8
        self.target_o = 384
        self.heatmap_method = "GAUSS"
        self.heatmap_sigma = 1.5
        self.offset_method = "GAUSS"
        self.offset_sigma = 1.0
        self.criterion_heatmap = 'l2'
        self.criterion_offset = 'cls'
        self.loss_heatmap_weight = 1
        self.loss_offset_weight = 0.05
        self.num_landmarks = 98
        self.use_gpu = True
        self.gpu_id = 0