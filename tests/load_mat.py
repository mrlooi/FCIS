import scipy.io as sio
m = sio.loadmat('./data/VOCdevkit/VOCSDS/cls/2008_007124.mat')

x = m['GTcls']['Segmentation'][0][0]
