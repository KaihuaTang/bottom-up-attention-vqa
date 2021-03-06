from __future__ import print_function

import errno
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn


EPS = 1e-7

def print_para(model):
    """
    Prints parameters of a model
    :param opt:
    :return:
    """
    st = {}
    strings = []
    total_params = 0
    for p_name, p in model.named_parameters():

        if not ('bias' in p_name.split('.')[-1] or 'bn' in p_name.split('.')[-1]):
            st[p_name] = ([str(x) for x in p.size()], np.prod(p.size()), p.requires_grad)
        total_params += np.prod(p.size())
    for p_name, (size, prod, p_req_grad) in sorted(st.items(), key=lambda x: -x[1][1]):
        strings.append("{:<50s}: {:<16s}({:8d}) ({})".format(
            p_name, '[{}]'.format(','.join(size)), prod, 'grad' if p_req_grad else '    '
        ))
    return '\n {:.1f}M total parameters \n ----- \n \n{}'.format(total_params / 1000000.0, '\n'.join(strings))

def optimistic_restore(network, state_dict):
    mismatch = False
    own_state = network.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print("Unexpected key {} in state_dict with size {}".format(name, param.size()))
            mismatch = True
        elif param.size() == own_state[name].size():
            own_state[name].copy_(param)
        else:
            print("Network has {} with size {}, ckpt has {}".format(name,
                                                                    own_state[name].size(),
                                                                    param.size()))
            mismatch = True

    missing = set(own_state.keys()) - set(state_dict.keys())
    if len(missing) > 0:
        print("We couldn't find {}".format(','.join(missing)))
        mismatch = True
    return not mismatch

def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)


def assert_array_eq(real, expected):
    assert (np.abs(real-expected) < EPS).all(), \
        '%s (true) vs %s (expected)' % (real, expected)


def load_folder(folder, suffix):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(suffix):
            imgs.append(os.path.join(folder, f))
    return imgs


def load_imageid(folder):
    images = load_folder(folder, 'jpg')
    img_ids = set()
    for img in images:
        img_id = int(img.split('/')[-1].split('.')[0].split('_')[-1])
        img_ids.add(img_id)
    return img_ids


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def weights_init(m):
    """custom weights initialization."""
    cname = m.__class__
    if cname == nn.Linear or cname == nn.Conv2d or cname == nn.ConvTranspose2d:
        m.weight.data.normal_(0.0, 0.02)
    elif cname == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        print('%s is not initialized.' % cname)


def init_net(net, net_file):
    if net_file:
        net.load_state_dict(torch.load(net_file))
    else:
        net.apply(weights_init)


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


class Logger(object):
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        print(msg)


class EvalbyTypeLogger(object):
    def __init__(self, a_type_dict, q_type_dict):
        self.a_type_dict = a_type_dict
        self.q_type_dict = q_type_dict
        self.at_num = len(a_type_dict)
        self.qt_num = len(q_type_dict)

        self.at_accu = np.zeros(self.at_num)
        self.at_count = np.zeros(self.at_num)
        self.qt_accu = np.zeros(self.qt_num)
        self.qt_count = np.zeros(self.qt_num)

    def update(self, score_tensor, a_type, q_type):
        """
        score_tensor: [batch_size, num_answers]
        a_type: [batch_size] LongTensor
        q_type: [batch_size] LongTensor
        """
        batch_scores = score_tensor.sum(1)
        a_type = a_type.view(-1)
        q_type = q_type.view(-1)

        for i in range(self.at_num):
            num_at_i = torch.nonzero(a_type == (i+1)).numel()
            self.at_count[i] += num_at_i
            score_at_i = ((a_type == (i+1)).float() * batch_scores).sum()
            self.at_accu[i] += score_at_i

        for i in range(self.qt_num):
            num_qt_i = torch.nonzero(q_type == (i+1)).numel()
            self.qt_count[i] += num_qt_i
            score_qt_i = ((q_type == (i+1)).float() * batch_scores).sum()
            self.qt_accu[i] += score_qt_i

    def printResult(self, show_q_type=False, show_a_type=True):
        if(show_a_type):
            print("========== Accuracy by Type of Answers ==========")
            for key in self.a_type_dict.keys():
                type_score = self.at_accu[self.a_type_dict[key]-1]
                type_num = self.at_count[self.a_type_dict[key]-1] + 1e-10
                print('Type: \t %s \t  Accuracy: \t %.6f \t Total Tpye Num: \t %.1f' % (key, float(type_score)/float(type_num), float(type_num)) )
        if(show_q_type):
            print("========== Accuracy by Type of Questions ==========")
            for key in self.q_type_dict.keys():
                type_score = self.qt_accu[self.q_type_dict[key]-1]
                type_num = self.qt_count[self.q_type_dict[key]-1] + 1e-10
                print('Type: \t %s \t  Accuracy: \t %.6f \t Total Tpye Num: \t %.1f' % (key, float(type_score)/float(type_num), float(type_num)) )
        #print("==================== End print ====================")


