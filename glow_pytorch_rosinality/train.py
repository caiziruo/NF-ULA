import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 3, 6, 7"
from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi

import argparse

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms, utils

from model import Glow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Glow trainer")
parser.add_argument("--batch", default=32, type=int, help="batch size")
parser.add_argument("--iter", default=200000, type=int, help="maximum iterations")
parser.add_argument("--n_flow", default=32, type=int, help="number of flows in each block")
parser.add_argument("--n_block", default=5, type=int, help="number of blocks")
parser.add_argument("--no_lu",action="store_true",help="use plain convolution instead of LU decomposed version")
parser.add_argument("--affine", action="store_true", help="use affine coupling instead of additive")
parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
parser.add_argument("--lr", default=5e-5, type=float, help="learning rate")
parser.add_argument("--n_channel", default=3, type=int, help="channels of image")
parser.add_argument("--img_size", default=128, type=int, help="image size")
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=16, type=int, help="number of samples")
parser.add_argument("path", metavar="PATH", type=str, help="Path to image directory")
parser.add_argument("--load_pt", default=211, type=int, help="load from checkpoint")
# parser.add_argument("--noise_on_data", default=0.01, type=float, help="noise on the training data")


def gradient_log_density(input, model):
    log_p, logdet, _ = model(input)

    gradients = torch.autograd.grad(outputs = log_p, inputs=input,
                        grad_outputs = torch.ones(log_p.size(), device = input.device),
                        create_graph = True, retain_graph = True, only_inputs = True)[0]

    return  gradients


def sample_data(path, batch_size, image_size):
    transform = transforms.Compose(
        [
            # transforms.CenterCrop(image_size),
            # transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)    
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=4
            )
            loader = iter(loader)
            yield next(loader)


def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def calc_loss(log_p, logdet, image_size, n_bins):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


def train(args, model, optimizer):
    dataset = iter(sample_data(args.path, args.batch, args.img_size))
    n_bins = 2.0 ** args.n_bits

    z_sample = []
    z_shapes = calc_z_shapes(args.n_channel, args.img_size, args.n_flow, args.n_block)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))

    if args.load_pt != 0:
        print('load from checkpoint %d'%(args.load_pt) )
        model.load_state_dict(torch.load( f"checkpoint/model_{str(args.load_pt + 1).zfill(6)}.pt" ))
        optimizer.load_state_dict(torch.load( f"checkpoint/optim_{str(args.load_pt + 1).zfill(6)}.pt" ))


    with tqdm(range(args.load_pt, args.iter)) as pbar:
        for i in pbar:
            image, _ = next(dataset)
            image = image.to(device)

            image = image * 255

            if args.n_bits < 8:
                image = torch.floor(image / 2 ** (8 - args.n_bits))

            image = image / n_bins - 0.5
            # noise_on_data = args.noise_on_data * torch.randn_like(image).to(device)

            if i == 0:
                with torch.no_grad():
                    log_p, logdet, _ = model.module(
                        image + torch.rand_like(image) / n_bins
                    )

                    continue

            else:
                # log_p, logdet, _ = model(noise_on_data + image + torch.rand_like(image) / n_bins)
                log_p, logdet, _ = model( image + torch.rand_like(image) / n_bins )

            logdet = logdet.mean()

            loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
            model.zero_grad()
            loss.backward()
            # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
            warmup_lr = args.lr
            optimizer.param_groups[0]["lr"] = warmup_lr
            optimizer.step()

            pbar.set_description(
                f"Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}"
            )

            if i % 1 == 0:
                with torch.no_grad():
                    utils.save_image(
                        model_single.reverse(z_sample).cpu().data,
                        f"sample/{str(i + 1).zfill(6)}.png",
                        normalize=True,
                        nrow=4,
                        value_range=(-0.5, 0.5),
                    )
                    # #################### Draw pdf of Glow.
                    # generated_images = model_single.reverse(z_sample)
                    # # generated_images = 0.5 * torch.randn( 50, 3, 128, 128 ).to(device)
                    # generated_images_1 = generated_images - 0.5
                    # # generated_images_1 = generated_images - 1
                    # generated_images_2 = generated_images + 0.5
                    # # generated_images_2 = generated_images + 1
                    # generated_images = torch.cat( (generated_images, generated_images_1, generated_images_2), dim = 0)
                    # x_norm = torch.sqrt(torch.sum( generated_images ** 2 , dim = (1, 2, 3) ))
                    # # x_norm = generated_images[:, 0, 44, 84]
                    # log_pdf, _, _ = model_single(generated_images)
                    # pdf = log_pdf
                    # # pdf = torch.exp(log_pdf)
                    # x_norm.pdf = Variable(x_norm), Variable(pdf)
                    # import matplotlib.pyplot as plt
                    # plt.scatter(x_norm.cpu().detach().data.numpy(), pdf.cpu().detach().data.numpy())
                    # plt.savefig("../pdf_Glow.png")
                    # time.sleep(100000)
                    # ####################

            if i % 2000 == 0:
                torch.save(
                    model.state_dict(), f"checkpoint/model_{str(i + 1).zfill(6)}.pt"
                )
                torch.save(
                    optimizer.state_dict(), f"checkpoint/optim_{str(i + 1).zfill(6)}.pt"
                )


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    model_single = Glow(
        args.n_channel, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )
    # total_params = sum(p.numel() for p in model_single.parameters())
    # print(total_params)
    model = nn.DataParallel(model_single)
    # model = model_single
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(args, model, optimizer)

