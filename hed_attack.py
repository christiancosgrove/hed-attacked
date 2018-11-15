from scipy.ndimage.filters import gaussian_filter
import numpy as np

#perturbation as per https://arxiv.org/abs/1803.06978
def perturb_image(image, diversity_prob=0.0):
    
    if np.random.uniform() < diversity_prob:

        _, height, width = image.shape

        resize_dims = (300, 300)

        rheight = np.random.randint(resize_dims[0], image.shape[1]) if resize_dims[0] < image.shape[1] else image.shape[1]
        rwidth = np.random.randint(resize_dims[1], image.shape[2]) if resize_dims[1] < image.shape[2] else image.shape[2]

        image = scipy.misc.imresize(image, (min(height, rheight), min(width, rwidth)))

        dheight = height - image.shape[0]
        dwidth = width - image.shape[1]

        pad_top = 0 if dheight == 0 else np.random.randint(dheight)
        pad_left = 0 if dwidth == 0 else np.random.randint(dwidth)

        image = np.pad(image, ((pad_top, dheight - pad_top), (pad_left, dwidth - pad_left), (0,0)), 'constant')
        image = image.transpose(2, 0, 1)
        
    return image

def forward(net, data, label):
    net.blobs['data'].reshape(1, *data.shape)
    net.blobs['data'].data[...] = data
    net.blobs['label'].reshape(1, 1, *label.shape)
    net.blobs['label'].data[...] = label
    return net.forward()


def attack(net, data, label, step, epsilon, mom, steps, smoothing, targeted=False):
    delta = np.zeros_like(data)
    accum = np.zeros_like(delta)
    for i in xrange(50):
        forward(net, perturb_image(data) + delta, label)
        net.backward(diffs=['data'])
        grad = np.squeeze(net.blobs['data'].diff)
        
        accum = mom * accum + grad
        
        if smoothing > 0:
        	accum = gaussian_filter(accum, smoothing)

        delta += (-1 if targeted else 1) * step * np.sign(accum)
        delta = np.clip(delta, -epsilon, epsilon)
        
    return data + delta