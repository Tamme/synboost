from PIL import Image
from PIL.ImageColor import getcolor, getrgb
from PIL.ImageOps import grayscale
import matplotlib.pyplot as plt

def falsecolor(src):
    if Image.isStringType(src):  # File path?
        src = Image.open(src)
    if src.mode not in ['L', 'RGB', 'RGBA']:
        raise TypeError('Unsupported source image mode: {}'.format(src.mode))
    src.load()

    # Create look-up-tables (luts) to map luminosity ranges to components
    # of the colors given in the color palette.

    cm = plt.get_cmap('viridis')
    palette = []
    for c in range(256):
        palette.append(tuple(i * 255 for i in cm(c)[:3]))
    
    luts = (tuple(c[0] for c in palette) +
            tuple(c[1] for c in palette) +
            tuple(c[2] for c in palette))

    # Create grayscale version of image of necessary.
    l = src if Image.getmodebands(src.mode) == 1 else grayscale(src)

    # Convert grayscale to an equivalent RGB mode image.
    if Image.getmodebands(src.mode) < 4:  # Non-alpha image?
        merge_args = ('RGB', (l, l, l))  # RGB version of grayscale.

    else:  # Include copy of src image's alpha layer.
        a = Image.new('L', src.size)
        a.putdata(src.getdata(3))
        luts += tuple(range(256))  # Add a 1:1 mapping for alpha values.
        merge_args = ('RGBA', (l, l, l, a))  # RGBA version of grayscale.

    # Merge all the grayscale bands back together and apply the luts to it.
    return Image.merge(*merge_args).point(luts)

def add_legend(img, legend):
    sz = img.size
    lsz = legend.size
    pcoords = ( (sz[0]-lsz[0]),0,sz[0],lsz[1] )
    img.paste(legend, pcoords)

if __name__ == '__main__':
    filename = 'results/anomaly/lost_found_example.png'

    t = Image.open(filename)
    img = falsecolor(t)
    legend = Image.open("legend.png")
    add_legend(img, legend)
    img.save("test.png")
