import numpy as np
import os
os.environ["GLOG_minloglevel"] = "2"
import sys
sys.path.insert(0,'/home/cheer/caffe/python')
import re
import caffe
import numpy as np
import skimage.io
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import vgg1024 as vgg
from collections import OrderedDict
import matplotlib.pyplot as plt
import cv2

base_path = os.getcwd()
model_path = os.path.join(base_path, 'caffemodel')
data_path = os.path.join(base_path,'data')

class CaffeParamProvider():
    def __init__(self, caffe_net):
        self.caffe_net = caffe_net

    def conv_kernel(self, name):
        k = self.caffe_net.params[name][0].data
        return k

    def conv_biases(self, name):
        k = self.caffe_net.params[name][1].data
        return k

    def bn_gamma(self, name):
        return self.caffe_net.params[name][0].data

    def bn_beta(self, name):
        return self.caffe_net.params[name][1].data

    def bn_mean(self, name):
        return (self.caffe_net.params[name][0].data/self.caffe_net.params[name][2].data)

    def bn_variance(self, name):
        return (self.caffe_net.params[name][1].data/self.caffe_net.params[name][2].data)

    def fc_weights(self, name):
        w = self.caffe_net.params[name][0].data
        return w

    def fc_biases(self, name):
        b = self.caffe_net.params[name][1].data
        return b


def preprocess(out):
    #"""Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
    #out = np.copy(img) * 255.0
    out = out[:, :, [2, 1, 0]]  # swap channel from RGB to BGR
    out[0] -= 104.008
    out[1] -= 116.669
    out[2] -= 122.675
    return out

def load_image(path, size=321):
    img = cv2.imread(path)
    resized_img = cv2.resize(img,(size,size)).astype(float)
    return resized_img

def load_caffe(img_p):

    prototxt = os.path.join(model_path,"deploy.prototxt")
    caffemodel = os.path.join(model_path,'train2_iter_8000.caffemodel')
    caffe.set_mode_gpu()
    net = caffe.Net(prototxt,caffemodel,caffe.TEST)
    return net

def parse_vgg16_1024(caffe_model,pth_vname,conv_num,conv_sub_num):

    if ('features' in pth_vname and 'weight' in pth_vname):

        print 'conv' + str(conv_num) + '_' + str(conv_sub_num)
        return  caffe_model.conv_kernel('conv' + str(conv_num) + '_' + str(conv_sub_num))

    if ('features' in pth_vname and 'bias' in pth_vname):

        print 'conv' + str(conv_num) + '_' + str(conv_sub_num)
        return  caffe_model.conv_biases('conv' + str(conv_num) + '_' + str(conv_sub_num))

    if ('classifier' in pth_vname and 'weight' in pth_vname):

        print 'fc' + str(conv_num)
        return   caffe_model.conv_kernel('fc' + str(conv_num))

    if ('classifier' in pth_vname and 'bias' in pth_vname):

        print 'fc' + str(conv_num)
        return caffe_model.conv_biases('fc' + str(conv_num))

def checkpoint_fn(layers):
    return 'resnet%d.pth' % layers

def convert(img_p):
    caffe_model = load_caffe(img_p)
    get_score_weights(caffe_model,'fc8_voc12')
    param_provider = CaffeParamProvider(caffe_model)
    model = vgg.vgg1024()
    old_dict = model.state_dict()
    new_state_dict = OrderedDict()
    keys = model.state_dict().keys()

    prenumber = 0
    conv_number = 1
    conv_sub_number = 1
    fc_prenumber = 0
    fc_number = 6

    for idx ,var_name in enumerate(keys[:]):

        if 'feature' in var_name:

           number = int(var_name[var_name.index('.') + 1: var_name.rindex('.')])
           if (number - prenumber) == 3:
              conv_number += 1
              conv_sub_number = 1
              prenumber = number

           data = parse_vgg16_1024(param_provider, var_name, conv_number, conv_sub_number)
           new_state_dict[var_name] = torch.from_numpy(data).float()

           if 'bias' in var_name:
              conv_sub_number += 1
              prenumber = number

        if 'classifier' in var_name:

            number = int(var_name[var_name.index('.') + 1: var_name.rindex('.')])
            if (number - fc_prenumber) == 3:
                fc_number += 1
                fc_prenumber = number

            if var_name == 'classifier.6.weight' or var_name == 'classifier.6.bias':
                if 'weight' in var_name:

                    fc_data = parse_vgg16_1024(param_provider, var_name, str(8) + '_voc12', conv_sub_num=0)
                else:
                    fc_data = parse_vgg16_1024(param_provider, var_name ,str(8) + '_voc12', conv_sub_num=0)
            else:
                   fc_data = parse_vgg16_1024(param_provider,var_name,fc_number,conv_sub_num = 0)
            new_state_dict[var_name] = torch.from_numpy(fc_data).float()

            if 'bias' in var_name:
                fc_prenumber = number

    for name, param in new_state_dict.items():
        if name not in old_dict:
            raise KeyError('unexpected key "{}" in state_dict'
                           .format(name))

        print('saving' + name)
        print param.size()
        if name == 'classifier.6.weight':
            param_fc8 = param.cpu().numpy()
        if name == 'classifier.6.bias':
            param_fc8 = param.cpu().numpy()

    model.load_state_dict(new_state_dict)

    torch.save(model.state_dict(),'data/deeplab_voc.pth')

def main():

    img = load_image(os.path.join(base_path,"data/cat.jpg"))
    img_p = preprocess(img)
    print "CONVERTING Multi-scale DeepLab_resnet"
    convert(img_p)

if __name__ == '__main__':
    main()

