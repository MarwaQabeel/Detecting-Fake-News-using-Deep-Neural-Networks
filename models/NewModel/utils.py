"""
Helper functions.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import math
import time
import random

# Pytorch packages
import torch
import torch.optim as optim
import torch.nn as nn

# Numpy
import numpy as np

# Tqdm progress bar
from tqdm import tqdm_notebook

RANDOM_SEED = 0


def set_seed():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


def set_seed_nb():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED + 1)


def deterministic_init(net: nn.Module):
    for p in net.parameters():
        if p.data.ndimension() >= 2:
            set_seed_nb()
            nn.init.xavier_uniform_(p.data)
        else:
            nn.init.zeros_(p.data)

def train(model, dataloader, optimizer, criterion, scheduler=None, device='cpu'):
    model.train()

    # Record total loss
    total_loss = 0.

    # Get the progress bar for later modification
    progress_bar = tqdm_notebook(dataloader, ascii=True)

    # Mini-batch training
    for batch_idx, data in enumerate(progress_bar):
        source = data[0].transpose(1, 0).to(device)
        target = data[1].transpose(1, 0).to(device)

        if model.__class__.__name__ == 'FullTransformerTranslator':
            translation = model(source, target)
        else:
            translation = model(source)
        translation = translation.reshape(-1, translation.shape[-1])
        target = target.reshape(-1)

        optimizer.zero_grad()
        loss = criterion(translation, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_description_str(
            "Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

    return total_loss, total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device='cpu'):
    # Set the model to eval mode to avoid weights update
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        # Get the progress bar
        progress_bar = tqdm_notebook(dataloader, ascii=True)
        for batch_idx, data in enumerate(progress_bar):
            source = data[0].transpose(1, 0).to(device)
            target = data[1].transpose(1, 0).to(device)

            if model.__class__.__name__ == 'FullTransformerTranslator':
                translation = model(source, target)
            else:
                translation = model(source)
            translation = translation.reshape(-1, translation.shape[-1])
            target = target.reshape(-1)

            loss = criterion(translation, target)
            total_loss += loss.item()
            progress_bar.set_description_str(
                "Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

    avg_loss = total_loss / len(dataloader)
    return total_loss, avg_loss


def unit_test_values(testcase):
    if testcase == 'rnn':
        return torch.FloatTensor([[-0.9080, -0.5639, -3.5862],
                                  [-1.2683, -0.4294, -2.6910],
                                  [-1.7300, -0.3964, -1.8972],
                                  [-2.3217, -0.4933, -1.2334]]), torch.FloatTensor([[0.9629,  0.9805, -0.5052,  0.8956],
                                                                                    [0.7796,  0.9508, -
                                                                                        0.2961,  0.6516],
                                                                                    [0.1039,  0.8786, -
                                                                                        0.0543,  0.1066],
                                                                                    [-0.6836,  0.7156,  0.1941, -0.5110]])

    if testcase == 'lstm':
        ht = torch.FloatTensor([[-0.0452,  0.7843, -0.0061,  0.0965],
                                [-0.0206,  0.5646, -0.0246,  0.7761],
                                [-0.0116,  0.3177, -0.0452,  0.9305],
                                [-0.0077,  0.1003,  0.2622,  0.9760]])
        ct = torch.FloatTensor([[-0.2033,  1.2566, -0.0807,  0.1649],
                                [-0.1563,  0.8707, -0.1521,  1.7421],
                                [-0.1158,  0.5195, -0.1344,  2.6109],
                                [-0.0922,  0.1944,  0.4836,  2.8909]])
        return ht, ct

    if testcase == 'encoder':
        expected_out = torch.FloatTensor([[[-0.7773, -0.2031],
                                         [-0.6186, -0.2321]],

                                        [[ 0.0599, -0.0151],
                                         [-0.9237,  0.2675]],

                                        [[ 0.6161,  0.5412],
                                         [ 0.7036,  0.1150]],

                                        [[ 0.6161,  0.5412],
                                         [-0.5587,  0.7384]],

                                        [[-0.9062,  0.2514],
                                         [-0.8684,  0.7312]]])
        expected_hidden = torch.FloatTensor([[[ 0.4912, -0.6078],
                                         [ 0.4932, -0.6244],
                                         [ 0.5109, -0.7493],
                                         [ 0.5116, -0.7534],
                                         [ 0.5072, -0.7265]]])
        return expected_out, expected_hidden

    if testcase == 'decoder':
        expected_out = torch.FloatTensor(
        [[-2.1507, -1.6473, -3.1772, -3.2119, -2.6847, -2.1598, -1.9192, -1.8130,
         -2.6142, -3.1621],
        [-1.9727, -2.1730, -3.3104, -3.1552, -2.4158, -1.7287, -2.1686, -1.7175,
         -2.6946, -3.2259],
        [-2.1952, -1.7092, -3.1261, -2.9943, -2.5070, -2.1580, -1.9062, -1.9384,
         -2.4951, -3.1813],
        [-2.1961, -1.7070, -3.1257, -2.9950, -2.5085, -2.1600, -1.9053, -1.9388,
         -2.4950, -3.1811],
        [-2.7090, -1.1256, -3.0272, -2.9924, -2.8914, -3.0171, -1.6696, -2.4206,
         -2.3964, -3.2794]])
        expected_hidden = torch.FloatTensor([[
                                            [-0.1854,  0.5561],
                                            [-0.6016,  0.0276],
                                            [ 0.0255,  0.3106],
                                            [ 0.0270,  0.3131],
                                            [ 0.9470,  0.8482]]])
        return expected_out, expected_hidden

    if testcase == 'seq2seq':
        expected_out = torch.FloatTensor(
        [[[-2.4136, -2.2861, -1.7145, -2.5612, -1.9864, -2.0557, -1.7461,
          -2.1898],
         [-2.0869, -2.9425, -2.0188, -1.6864, -2.5141, -2.3069, -1.4921,
          -2.3045]],

        [[-1.8506, -2.3783, -2.1297, -1.9083, -2.5922, -2.3552, -1.5708,
          -2.2505],
         [-2.0939, -2.1570, -2.0352, -2.2691, -2.1251, -1.8906, -1.8156,
          -2.3654]]]
        )
        return expected_out

    if testcase == 'attention':

        hidden = torch.FloatTensor(
            [[[-0.7232, -0.6048],
              [0.9299, 0.7423],
              [-0.4391, -0.7967],
              [-0.0012, -0.2803],
              [-0.3248, -0.3771]]]
        )

        enc_out = torch.FloatTensor(
            [[[-0.7773, -0.2031],
              [-0.6186, -0.2321]],

             [[0.0599, -0.0151],
              [-0.9237, 0.2675]],

             [[0.6161, 0.5412],
              [0.7036, 0.1150]],

             [[0.6161, 0.5412],
              [-0.5587, 0.7384]],

             [[-0.9062, 0.2514],
              [-0.8684, 0.7312]]]
        )

        expected_attention = torch.FloatTensor(
            [[[0.4902, 0.5098]],

             [[0.7654, 0.2346]],

             [[0.4199, 0.5801]],

             [[0.5329, 0.4671]],

             [[0.6023, 0.3977]]]
        )
        return hidden, enc_out, expected_attention

    if testcase == 'seq2seq_attention':
        expected_out = torch.FloatTensor(
            [[[-2.8071, -2.4324, -1.7512, -2.7194, -1.7530, -2.1202, -1.6578,
               -2.0519],
              [-2.2137, -2.4308, -2.0972, -2.1079, -1.9882, -2.0411, -1.6965,
               -2.2229]],

             [[-1.9549, -2.4265, -2.1293, -1.9744, -2.2882, -2.4210, -1.4311,
               -2.4892],
              [-2.1284, -2.2369, -2.1940, -1.9027, -2.1065, -2.2274, -1.7391,
               -2.2220]]]
        )
        return expected_out

    if testcase == 'full_trans_fwd':
        expected_out = torch.FloatTensor(
            [[[0.0778, 0.7265],
              [-0.2000, 0.4644],
              [0.6747, -0.0823],
              [-0.5711, 0.0069],
              [-0.2169, -0.1791],
              [0.2986, 0.0047],
              [-0.1547, 0.0963],
              [-0.2114, 1.0473],
              [-0.0181, 0.1785],
              [-0.0417, -0.1791],
              [0.6582, 0.0339],
              [0.3202, -0.2229],
              [0.2652, 0.0337],
              [-0.4046, 0.4451],
              [-0.5675, 0.5339],
              [0.4556, 0.4203],
              [0.2243, -0.5047],
              [0.3853, 0.6743],
              [0.5357, 0.2614],
              [0.5677, -0.2949],
              [0.2759, 0.1137],
              [0.3195, 0.5535],
              [-0.0943, -0.3804],
              [0.3408, -0.0497],
              [1.2820, 0.0533],
              [0.0381, -0.1676],
              [0.4236, 0.4226],
              [0.6988, 0.0316],
              [0.1185, 0.3314],
              [0.9255, -0.1155],
              [0.6882, 0.1838],
              [0.7028, 0.0494],
              [-0.0224, 0.6315],
              [0.6731, 0.4730],
              [1.1470, -0.3827],
              [0.7051, -0.5507],
              [0.7544, 0.5742],
              [0.7893, 0.4422],
              [0.4424, -0.4766],
              [0.2874, -0.3799],
              [0.6133, 0.4561],
              [0.4552, 0.3703],
              [0.4266, -0.0648]],

             [[0.0937, -0.7343],
              [-0.4760, 0.3398],
              [0.5276, 0.1216],
              [-0.0095, -0.1599],
              [-0.6845, 0.4819],
              [0.6382, -0.3006],
              [0.2553, -0.8615],
              [-0.3949, 0.0186],
              [0.2150, 0.3198],
              [-0.0346, -0.0189],
              [0.0281, -0.4183],
              [0.1553, 0.3138],
              [0.2557, -0.3189],
              [-0.2573, -0.4039],
              [-0.2927, -0.3627],
              [0.4587, -0.2063],
              [0.6203, -0.6603],
              [0.6789, 0.4088],
              [0.4059, -0.3198],
              [0.9528, -0.4760],
              [0.2045, -0.2409],
              [0.6001, 0.0450],
              [0.3609, -0.7221],
              [0.5056, 0.3740],
              [1.2870, -0.0250],
              [0.3148, -0.3431],
              [0.1315, 0.6088],
              [-0.2787, 0.2217],
              [0.0385, 0.1709],
              [0.5690, -0.9050],
              [0.0580, 0.1704],
              [-0.0131, 0.2116],
              [-0.2518, 0.4717],
              [1.0309, -0.5988],
              [1.1688, -0.4980],
              [0.6490, 0.1826],
              [0.7413, 0.5611],
              [0.6591, 0.8001],
              [0.3364, -0.3319],
              [0.5703, -0.0281],
              [0.0165, 0.2276],
              [1.2226, -0.0891],
              [0.1480, -0.2047]]]
        )
        return expected_out

    if testcase == 'full_trans_translate':
        expected_out = torch.FloatTensor(
            [[[1.0595e-01, 1.8132e-01],
              [-6.8386e-02, -5.7950e-01],
              [4.3969e-01, 9.4059e-01],
              [-3.5763e-01, 4.7057e-01],
              [5.4296e-01, -1.1629e-01],
              [7.2519e-01, -4.4772e-01],
              [2.1273e-01, -2.2399e-01],
              [2.7755e-01, 3.4979e-01],
              [3.5568e-01, -2.8527e-02],
              [-1.3862e-01, -2.4950e-02],
              [5.8719e-01, -2.0228e-01],
              [-6.6054e-01, -5.3559e-02],
              [8.1099e-02, -3.8780e-01],
              [-3.9600e-01, 2.4079e-01],
              [-2.9146e-01, 4.4521e-01],
              [4.9296e-01, 4.0123e-01],
              [1.5432e-01, -2.5274e-01],
              [1.2068e+00, 4.9792e-01],
              [2.3705e-01, -4.1174e-01],
              [5.1984e-01, -1.7247e-01],
              [1.0611e+00, -1.6193e-01],
              [1.0168e+00, 5.8163e-01],
              [2.0323e-01, -7.4506e-01],
              [-3.0458e-02, 2.8247e-01],
              [1.1358e+00, 6.1935e-02],
              [7.2704e-01, -4.1770e-01],
              [6.4098e-01, 3.4414e-01],
              [-1.8323e-01, -1.3036e-01],
              [3.4393e-01, 2.0912e-01],
              [7.2846e-01, -1.0274e+00],
              [2.0891e-01, -3.2622e-02],
              [1.7913e-01, 5.1587e-01],
              [2.0062e-01, 1.3634e-01],
              [5.6860e-02, -2.6757e-01],
              [7.4872e-01, -7.2282e-01],
              [1.0721e+00, -9.0005e-01],
              [1.3052e+00, 7.0182e-01],
              [3.3806e-01, 2.3070e-01],
              [2.5809e-01, -7.9196e-01],
              [4.6949e-02, 2.4227e-01],
              [9.9623e-01, 2.7459e-01],
              [6.1486e-01, 4.5702e-01],
              [4.4957e-01, 3.0312e-01]],

             [[-3.0908e-01, -2.5219e-01],
              [-2.6741e-01, 5.7882e-01],
              [8.3114e-02, 7.0994e-01],
              [-3.6043e-01, 3.8229e-01],
              [4.5771e-01, 4.0397e-01],
              [2.5784e-01, 1.0572e+00],
              [-7.9399e-02, -2.2311e-01],
              [6.0587e-01, 3.1288e-01],
              [5.8429e-01, 6.4703e-01],
              [4.2943e-01, 7.7081e-01],
              [9.3079e-01, -1.3188e-01],
              [-2.3681e-01, 9.7715e-02],
              [1.8777e-01, 3.1237e-02],
              [-1.2459e-01, 8.7308e-02],
              [-6.0669e-01, -8.7146e-02],
              [7.4455e-01, -2.9029e-01],
              [3.6704e-01, -5.9803e-01],
              [3.1790e-01, 9.4313e-01],
              [7.6396e-01, -8.3055e-01],
              [7.0670e-01, -2.7663e-01],
              [6.9172e-01, 1.6298e-01],
              [3.5136e-01, 4.6048e-01],
              [-3.9346e-02, -5.8288e-01],
              [4.4697e-01, 5.1974e-01],
              [1.0146e+00, 2.1931e-01],
              [5.3961e-01, -2.7246e-01],
              [8.3931e-02, 5.8164e-01],
              [2.7313e-01, -4.3820e-01],
              [-2.4355e-01, -7.0763e-04],
              [7.5250e-01, -6.1899e-01],
              [-3.7182e-01, -4.4931e-01],
              [-3.6951e-01, 4.8502e-01],
              [5.1141e-01, 3.5145e-01],
              [3.7258e-01, -1.4006e-01],
              [1.0030e+00, -6.4444e-01],
              [1.2116e+00, -2.4928e-02],
              [7.9971e-01, 2.4983e-02],
              [5.8640e-01, 2.6060e-01],
              [5.5314e-01, -3.2189e-01],
              [3.7423e-01, -4.2277e-01],
              [4.2615e-01, 5.7060e-01],
              [6.5556e-01, 3.7038e-01],
              [5.8625e-01, -7.8578e-01]]]
        )
        return expected_out



