# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import dnnlib
import legacy
from hashlib import sha256
from flask import send_file
import gc 

def breed_waifu(
    G,
    *,
    num_steps                  = 5,
    w_avg_samples              = 1,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    verbose                    = False,
    device                     : torch.device,
    seed1                      = "test1",
    seed2                      = "test2",
    outdir                     = "out_breed"
):

    def logprint(*args):
        if verbose:
            print(*args)

    #Clearing cache
    gc.collect()
    torch.cuda.empty_cache() 
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    w_avg, w_std = compute_w_stat(G, device, w_avg_samples, seed1)
    w2_avg, w2_std = compute_w_stat(G, device, w_avg_samples, seed2)

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    w2_opt = torch.tensor(w2_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    w2_out = torch.zeros([num_steps] + list(w2_opt.shape[1:]), dtype=torch.float32, device=device)
    
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)
    optimizer2 = torch.optim.Adam([w2_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = noise_scale(w_std, initial_noise_factor, t, noise_ramp_length)
        w2_noise_scale = noise_scale(w2_std, initial_noise_factor, t, noise_ramp_length)
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode='const')

        # Synth images from opt_w2.
        w2_noise = torch.randn_like(w2_opt) * w2_noise_scale
        w2s = (w2_opt + w2_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images2 = G.synthesis(w2s, noise_mode='const')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Downsample image2 to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images2 = (synth_images2 + 1) * (255/2)
        if synth_images2.shape[2] > 256:
            synth_images2 = F.interpolate(synth_images2, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        synth_features2 = vgg16(synth_images2, resize_images=False, return_lpips=True)
        dist = (synth_features2 - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    # Save final projected frame and W vector.
    os.makedirs(outdir, exist_ok=True)

    # Save parent1
    # projected_w = projected_w_steps[0]
    # synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    # synth_image = (synth_image + 1) * (255/2)
    # synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    # PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/parent1.png')
    #np.savez(f'{outdir}/projected_w_parent1.npz', w=projected_w.unsqueeze(0).cpu().numpy())

    # Save parent2
    # w2_noise_scale = w2_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
    # w2_noise = torch.randn_like(w2_opt) * w2_noise_scale
    # w2s = (w2_opt + w2_noise).repeat([1, G.mapping.num_ws, 1])
    # synth_image2 = G.synthesis(w2s, noise_mode='const')
    # synth_image2 = (synth_image2 + 1) * (255/2)
    # synth_image2 = synth_image2.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    # PIL.Image.fromarray(synth_image2, 'RGB').save(f'{outdir}/parent2.png')

    projected_w_steps = w_out.repeat([1, G.mapping.num_ws, 1])
    projected_w = projected_w_steps[-1]
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    child_path = f'{outdir}/{seed1}_{seed2}_{str(num_steps)}.png'
    PIL.Image.fromarray(synth_image, 'RGB').save(child_path)
    #np.savez(f'{outdir}/projected_w_child.npz', w=projected_w.unsqueeze(0).cpu().numpy())

    gc.collect()
    torch.cuda.empty_cache()

    return send_file(child_path, mimetype='image/png')

#----------------------------------------------------------------------------

def compute_w_stat(G, device, w_avg_samples, seed):
    hash = sha256(seed.encode('utf-8'))
    seed1_digest = np.frombuffer(hash.digest(), dtype='uint32')
    z_samples = np.random.RandomState(seed1_digest).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5
    return w_avg, w_std

def noise_scale(w_std, initial_noise_factor, t, noise_ramp_length):
    return w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2