
class ConfigLight():
    #needed by spade resynth
    def __init__(self, use_spade=False):
        self.arch='image_segmentation_icnet.libs.models.ICNet'
        self.aspect_ratio=2.0
        self.batchSize=1
        self.cache_filelist_read=False
        self.cache_filelist_write=False
        self.checkpoints_dir='models'
        self.contain_dontcare_label=False
        self.crop_size=512
        self.dataroot='./results/temp'
        self.dataset_mode='cityscapes'
        self.demo_folder='./sample_images'
        self.display_winsize=512
        self.dist_url='tcp://10.1.72.171:8000'
        self.gpu=0
        self.how_many=1000
        self.idx_server=0
        self.init_type='xavier'
        self.init_variance=0.02
        self.isTrain=False
        self.label_nc=35
        self.load_from_opt_file=False
        self.load_size=512
        self.max_dataset_size=9223372036854775807
        self.model='pix2pix'
        self.mpdist=False
        self.nThreads=0
        self.name='image-synthesis'
        self.nef=16
        self.netG='condconv'
        self.ngf=64
        self.ngpus_per_node=8
        self.no_flip=True
        self.no_instance=False
        self.no_pairing_check=False
        self.no_segmentation=False
        self.norm_D='spectralinstance'
        self.norm_E='spectralinstance'
        self.norm_G='spectralsync_batch'
        self.num_servers=1
        self.output_nc=3
        self.phase='test'
        self.preprocess_mode='fixed'
        self.results_dir='./results/'
        self.semantic_nc=36
        self.serial_batches=True
        self.snapshot='models/image-segmentation/icnet_final.pth'
        self.use_vae=True
        self.which_epoch='latest'
        self.z_dim=256
    
        if use_spade:
            self.num_upsampling_layers='more'  #more, normal  
            self.gpu_ids=[0]
            self.norm_G='spectralspadesyncbatch3x3'
            self.use_vae=False
            self.name='image-synthesis_spade'
            self.netG='spade' #spade or pix2pixhd or condconv
    
    def __str__(self):
        return ', '.join("%s: %s" % item for item in vars(self).items())
    
    def __repr__(self):
        return __str__(self)
