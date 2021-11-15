import argparse
import os
from PIL import Image
import numpy as np
import time
import cv2
from estimator import AnomalyDetector, ICNET, DEEPLAB, SPADE, CCFPSE
import colormap_helper

# function for segmentations
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
           153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
           255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    """
    Colorize a segmentation mask.
    """
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

parser = argparse.ArgumentParser()
parser.add_argument('--demo_folder', type=str, default='./sample_images', help='Path to folder with images to be run.')
parser.add_argument('--save_folder', type=str, default='./results', help='Folder to where to save the results')
parser.add_argument('--verbose_save', type=bool, default=False, help='Save each output separately')
parser.add_argument('--fp16', type=bool, default=False, help='Do mixed precision calculations')
opts = parser.parse_args()
save_folder = opts.save_folder

images = [os.path.join(opts.demo_folder, image) for image in os.listdir(opts.demo_folder)]

DETECTOR = DEEPLAB
SYNTHESIZER = CCFPSE
if "icnet" in save_folder.lower():
    DETECTOR = ICNET
if "spade" in save_folder.lower():
    SYNTHESIZER = SPADE

detector = AnomalyDetector(ours=True, detector=DETECTOR, synthesizer=SYNTHESIZER, fp16=opts.fp16)

# Save folders
semantic_path = os.path.join(save_folder, 'semantic')
anomaly_path = os.path.join(save_folder, 'anomaly')
synthesis_path = os.path.join(save_folder, 'synthesis')
entropy_path = os.path.join(save_folder, 'entropy')
distance_path = os.path.join(save_folder, 'distance')
perceptual_diff_path = os.path.join(save_folder, 'perceptual_diff')
concatenated_path = os.path.join(save_folder, 'concatenated')

if opts.verbose_save:
    os.makedirs(semantic_path, exist_ok=True)
    os.makedirs(anomaly_path, exist_ok=True)
    os.makedirs(synthesis_path, exist_ok=True)
    os.makedirs(entropy_path, exist_ok=True)
    os.makedirs(distance_path, exist_ok=True)
    os.makedirs(perceptual_diff_path, exist_ok=True)
os.makedirs(concatenated_path, exist_ok=True)

legend_path = "legend.png"
legend_exists = os.path.exists(legend_path)
if legend_exists:
    legend = Image.open(legend_path)

for idx, image_path in enumerate(images):
    basename = os.path.basename(image_path).replace('.jpg', '.png')
    print('Evaluating image %i out of %i'%(idx+1, len(images)))
    image = Image.open(image_path)
    #image_cv2 = cv2.imread(image_path, cv2.IMREAD_COLOR)
    t0 = time.time()
    results = detector.estimator_image(image)
    t1 = time.time()
    print("Time {}".format(t1 - t0))

    basename = os.path.basename(image_path).replace('.png', '.jpg')
    anomaly_map = results['anomaly_map'].convert('RGB')

    anomaly_map = colormap_helper.falsecolor(anomaly_map)
    if legend_exists:
        colormap_helper.add_legend(anomaly_map, legend)

    if opts.verbose_save:
        anomaly_map.save(os.path.join(anomaly_path,basename))

    semantic_map = colorize_mask(np.array(results['segmentation'])).convert('RGB')
    if opts.verbose_save:
        semantic_map.save(os.path.join(semantic_path,basename))

    synthesis = results['synthesis'].convert('RGB')
    if opts.verbose_save:
        synthesis.save(os.path.join(synthesis_path,basename))

    softmax_entropy = results['softmax_entropy'].convert('RGB')
    if opts.verbose_save:
        softmax_entropy.save(os.path.join(entropy_path,basename))

    softmax_distance = results['softmax_distance'].convert('RGB')
    if opts.verbose_save:
        softmax_distance.save(os.path.join(distance_path,basename))

    perceptual_diff = results['perceptual_diff'].convert('RGB')
    if opts.verbose_save:
        perceptual_diff.save(os.path.join(perceptual_diff_path,basename))
    
    w,h = anomaly_map.size
    concat_img = Image.new('RGB', (2*w,4*h))
    concat_img.paste(image, (0,0))
    concat_img.paste(semantic_map, (w,0))
    concat_img.paste(synthesis, (0,h))
    concat_img.paste(anomaly_map, (w,h))
    concat_img.paste(softmax_entropy, (0,2*h))
    concat_img.paste(softmax_distance, (w,2*h))
    concat_img.paste(perceptual_diff, (0,3*h))
    
    from PIL import ImageDraw, ImageFont
    color = (57, 255, 20)
    font = ImageFont.truetype("opensans.ttf", 62)
    draw = ImageDraw.Draw(concat_img)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    draw.text((0, 0), 'Original', color, font)
    draw.text((w, 0), 'Segmented by {}'.format(DETECTOR), color, font)
    draw.text((0, h), '{} GAN synthesis'.format(SYNTHESIZER), color, font)
    draw.text((w, h), 'Anomaly map {}'.format('with prior' if detector.prior else ": no prior"), color, font)
    draw.text((0, 2*h), 'Softmax entropy', color, font)
    draw.text((w, 2*h), 'Softmax distance', color, font)
    draw.text((0, 3*h), 'Perceptual diff', color, font)
    draw.text((w, 3*h), results['text'], color, font)

    concat_img.save(os.path.join(concatenated_path,basename))
