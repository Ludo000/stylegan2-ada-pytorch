import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

import generate



import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

from hashlib import sha256
from flask import send_file
from breed import breed_waifu
from utils import represents_int

app = Flask(__name__)
CORS(app, support_credentials=True)


with open('config.json') as json_file:
    data = json.load(json_file)
    device = torch.device(data['device'])
    network_pkl = data['network_pkl']
    outdir = data['outdir']
    print('Loading networks from "%s"...' % network_pkl)
    os.makedirs(outdir, exist_ok=True)
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
label = torch.zeros([1, G.c_dim], device=device)

@app.route('/generate', methods=['GET'])
@cross_origin(supports_credentials=True)
def endpoint_generate_waifu():
    if request.method == 'GET':
        return generate_waifu(
            request.args.get('seed'),
            1,
            'out',
            'const',
            data['device']
        )

@app.route('/breed', methods=['GET'])
@cross_origin(supports_credentials=True)
def endpoint_breed_waifu():
    if request.method == 'GET':
        seed1 = request.args.get('seed1')
        seed2 = request.args.get('seed2')
        step = request.args.get('step')
        if(step is None or not represents_int(step)):
            step = "5"
        i_step = int(step)
        if(i_step < 1):
            i_step = 1
        elif(i_step > 100):
            i_step = 100
        img_path = f'out_breed/{seed1}_{seed2}_{i_step}.png'
        try:
            PIL.Image.open(img_path)
            return send_file(img_path, mimetype='image/png')
        except FileNotFoundError:
            return breed_waifu(
                G                          = G,
                num_steps                  = i_step,
                w_avg_samples              = 1,
                initial_learning_rate      = 0.1,
                initial_noise_factor       = 0.05,
                lr_rampdown_length         = 0.25,
                lr_rampup_length           = 0.05,
                noise_ramp_length          = 0.75,
                regularize_noise_weight    = 1e5,
                verbose                    = False,
                device                     = device,
                seed1                      = seed1,
                seed2                      = seed2,
                outdir                     = "out_breed"
            )

def generate_waifu(
    seed: str,
    truncation_psi: float,
    outdir: str,
    noise_mode: str,
    device: str
):
    if seed is not None:
        img_path = f'{outdir}/{seed:s}.png'
        try:
            PIL.Image.open(img_path)
        except FileNotFoundError:

            print('Generating image for seed %s ...' % (seed))
            hash = sha256(seed.encode('utf-8'))
            seed_digest = np.frombuffer(hash.digest(), dtype='uint32')
            z = torch.from_numpy(np.random.RandomState(seed_digest).randn(1, G.z_dim)).to(device)
            if(device == 'cpu'):
                img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode, force_fp32=True)
            else:
                img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(img_path)

        return send_file(img_path, mimetype='image/png')



if __name__ == '__main__':
    app.run(host= '0.0.0.0', port="3000")
