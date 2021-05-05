import itertools
from multiprocessing.dummy import freeze_support
import glob
import os
import torch.nn as nn
import torch
import Dataloaders
import cycle_models
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pylab as plt
from PIL import Image
from torch.utils.data import DataLoader
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pylab as plt
import numpy as np
import  GPUtil



LAMBDA = 10
LAMBDA_IDENTITY = 0.5
EPOCHS = 200
ADAM_BETA = 0.5
START_LEARNING_RATE = 0.0002

#source_data = Dataloaders.DataLoader()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Losses
loss_inp_distr = nn.BCELoss()


def main():
    image_size = (128, 128)
    device = 'cuda'
    batch = 2
    downscale = 2

    torch.cuda.empty_cache()

    # Delete old lr batches in Folder
    for file in os.listdir('D:\DataSets\Downscaled_HR'):
        if file.endswith('.png'):
            os.remove('D:\DataSets\Downscaled_HR\\'+str(file))

    # Delete old generated images in Folder
    for file in os.listdir('D:\DataSets\Cycle_outputs'):
        if file.endswith('.png'):
            os.remove('D:\DataSets\Cycle_outputs\\'+str(file))

            # Load HR and LR Images
    lr_source_batch, lr_data = Dataloaders.LR_Source_Dataloader(image_size, device, batch)
    hr_source_batch, hr_data = Dataloaders.HR_Source_Dataloader(image_size, device, batch)

    # Create LR Image from HR Image
    transform_to_lowres = transforms.Resize((image_size[0] // downscale, image_size[0] // downscale),
                      interpolation=transforms.InterpolationMode.BICUBIC)
    downscaled_hr_batch = transform_to_lowres(hr_source_batch)
    # save them in folder to be able to load with DataLoader
    for i in range(len(downscaled_hr_batch)):
        save_image(downscaled_hr_batch[i],"D:\DataSets\Downscaled_HR\lr"+str(i)+".png")
    # load downscaled images
    downscaled_data = Dataloaders.Downscaled_DataLoader(device, batch)



    #torch.cuda.empty_cache()

    G_GEN_noise_domain = cycle_models.Cycle_Generator().to(device)
    F_GEN_bicubic_domain = cycle_models.Cycle_Generator().to(device)
    Z_DISC_bicubic = cycle_models.Cycle_Discriminator().to(device)
    X_DISC_noise = cycle_models.Cycle_Discriminator().to(device)

    train(G_GEN_noise_domain, F_GEN_bicubic_domain, Z_DISC_bicubic, X_DISC_noise, EPOCHS, lr_data, downscaled_data, device)



def train(G_GEN_noise_domain, F_GEN_bicubic_domain, Z_DISC_bicubic, X_DISC_noise, e_cnt, lr_dataloader, dowscaled_hr_dataloader, device):
    torch.cuda.empty_cache()

    Criterion_GAN = nn.MSELoss().to(device)
    Criterion_Cycle = nn.L1Loss().to(device)
    Criterion_Identity = nn.L1Loss().to(device)
    Loss_G_GEN = []
    Loss_F_GEN = []
    #Loss_D_bicubic_real = []
    #Loss_D_bicubic_fake = []
    Loss_D_bicubic = []
    Loss_D_noise = []

    imgs = []

    # Adam Optimizers
    optimizer_generators = torch.optim.Adam(itertools.chain(F_GEN_bicubic_domain.parameters(), G_GEN_noise_domain.parameters()), lr=START_LEARNING_RATE, betas=(ADAM_BETA, 0.999))
    optimizer_disc = torch.optim.Adam(itertools.chain(Z_DISC_bicubic.parameters(), X_DISC_noise.parameters()), lr= START_LEARNING_RATE, betas=(ADAM_BETA, 0.999))


    for e in range(e_cnt):
        print("start Epoch"+str(e))
        #(lr_d, dowscaled_d)
        for i, data in enumerate(zip(lr_dataloader, dowscaled_hr_dataloader)):
            print("Number of iterations: " + str(i))
            print(data[0].shape)
            # real input
            f_real_input = data[0].to(device)
            g_real_input = data[1].to(device)

            #tmp=nn.ReflectionPad2d(3)
            #out = tmp(f_real_input)
            #print(out.shape)

            # noise -> downsampled -> noise
            # noise -> downsampled'
            optimizer_generators.zero_grad()

            f_generated = F_GEN_bicubic_domain(f_real_input) # noise -> F -> bicubic'
            print(f_generated.shape)
            f_prediction = Z_DISC_bicubic(f_generated.detach()) # detachen oder nicht????
            #loss_F = loss_gen_cycle_input_distr(prediction)
            #print(f_prediction[1].shape)
            loss_F_gen = Criterion_GAN(f_prediction, torch.ones_like(f_prediction)) # zeros like oder ones like verwenden??

            # dowsampled' -> noise''
            # cycle consistency
            g_generated = G_GEN_noise_domain(f_generated)
            # calc loss between input noise image and gen noise'' image
            loss_G_gen = Criterion_Cycle(g_generated, f_real_input)*LAMBDA

            # identity loss
            identity_gen_F = F_GEN_bicubic_domain(g_real_input)
            loss_identity_F = Criterion_Identity(identity_gen_F, g_real_input) * LAMBDA

            # Summ loss of F (Downsample GAN)
            loss_F = loss_F_gen + loss_G_gen + loss_identity_F
            loss_F.backward(retain_graph = False)

            optimizer_generators.step()
            Loss_F_GEN.append(loss_F)

            # downsampled -> noise' -> downsampled''
            # downsampled -> noise'
            optimizer_generators.zero_grad()

            g_2ndcycle_generated = G_GEN_noise_domain(g_real_input)
            g_2ndcycle_prediction = X_DISC_noise(g_2ndcycle_generated.detach())
            loss_G_2ndcycle_gen = Criterion_GAN(g_2ndcycle_prediction, torch.ones_like(g_2ndcycle_prediction))

            # noise' -> downsampled''
            f_2ndcycle_generated = F_GEN_bicubic_domain(g_2ndcycle_generated)
            # calc loss between input bicubic downsampled image and generated downsampled'' img
            loss_F_2ndcycle_gen = Criterion_Cycle(f_2ndcycle_generated, g_real_input)*LAMBDA
            GPUtil.showUtilization()
            # identity loss
            identity_gen_G = G_GEN_noise_domain(f_real_input)
            loss_identity_G = Criterion_Identity(identity_gen_G, f_real_input) * LAMBDA * LAMBDA_IDENTITY

            # Sum loss of G (Noise GAN)
            loss_G = loss_G_2ndcycle_gen + loss_F_2ndcycle_gen + loss_identity_G
            loss_G.backward(retain_graph = False)
            optimizer_generators.step()

            Loss_G_GEN.append(loss_G)


            # Discriminators
            # *** Discriminator for bicubic downsampled images ***
            optimizer_disc.zero_grad()
            # disc for real bicubic imgs
            bicubic_disc_real_prediction = Z_DISC_bicubic(g_real_input)
            loss_bicubic_disc_real = Criterion_GAN(bicubic_disc_real_prediction, torch.ones_like(bicubic_disc_real_prediction))

            # disc for fake bicubic imgs
            bicubic_disc_fake_prediction = Z_DISC_bicubic(f_generated.detach())
            loss_bicubic_disc_fake = Criterion_GAN(bicubic_disc_fake_prediction, torch.zeros_like(bicubic_disc_fake_prediction))

            # disc bicubic loss
            loss_bicubic_disc = (loss_bicubic_disc_fake + loss_bicubic_disc_real) * 0.5
            loss_bicubic_disc.backward()
            optimizer_disc.step()
            Loss_D_bicubic.append(loss_bicubic_disc)

            # *** Discriminator for noise images ***
            optimizer_disc.zero_grad()
            # disc for real noise imgs
            noise_disc_real_prediction = X_DISC_noise(f_real_input)
            loss_noise_disc_real = Criterion_GAN(noise_disc_real_prediction, torch.ones_like(noise_disc_real_prediction))

            # disc for fake noise imgs
            noise_disc_fake_prediction = X_DISC_noise(g_2ndcycle_generated.detach())
            loss_noise_disc_fake = Criterion_GAN(noise_disc_fake_prediction, torch.zeros_like(noise_disc_fake_prediction))

            # disc noise loss
            loss_noise_disc = (loss_noise_disc_fake + loss_noise_disc_real) * 0.5
            loss_noise_disc.backward()
            optimizer_disc.step()
            Loss_D_noise.append(loss_noise_disc)

            if i % 10 == 0:
                path = 'D:\DataSets\Cycle_outputs\\'
                save_image(f_generated, path+'f_gen'+str(e)+'.png')
                save_image(g_2ndcycle_generated, path + 'g2nd_gen' + str(e) + '.png')
                save_image(f_2ndcycle_generated, path + 'f2nd_gen' + str(e) + '.png')
            print("done")



        print("Epoch "+str(e)+" finished")

if __name__ == '__main__':
   #freeze_support()
   main()




