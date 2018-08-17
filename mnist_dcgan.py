import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

compose = transforms.Compose(
        [transforms.Scale(32),
         transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])

train_data = datasets.MNIST(root="./mnist_data/", train=True, transform=compose, download=True)
test_data = datasets.MNIST(root="./mnist_data/", train=False, transform=compose, download=True)

train_load = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
test_load = DataLoader(dataset=test_data, batch_size=16, shuffle=False)

def display(img_batch):
	img = (img_batch[0].view(1, 32, 32)).data.numpy()
	img = cv2.resize(np.squeeze(img), (100, 100))
	cv2.imshow("Test", img)
	cv2.waitKey(1)

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()	#input 32x32
		self.layer1 = nn.Sequential(
						nn.Conv2d(1, 128, 4, 2, 1),	#16x16
						nn.LeakyReLU(0.2)
						)
		self.layer2 = nn.Sequential(
						nn.Conv2d(128, 256, 4, 2, 1),
						nn.BatchNorm2d(256),
						nn.LeakyReLU(0.2)
						)
		self.layer3 = nn.Sequential(
						nn.Conv2d(256, 512, 4, 2, 1),
						nn.BatchNorm2d(512),
						nn.LeakyReLU(0.2)
						)
		self.layer4 = nn.Sequential(
						nn.Conv2d(512, 1, 4, 1),
						nn.Sigmoid()
						)

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		return x

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.layer1 = nn.Sequential(
						nn.ConvTranspose2d(100, 512, 4, 1),
						nn.BatchNorm2d(512),
						nn.ReLU()
						)
		self.layer2 = nn.Sequential(
						nn.ConvTranspose2d(512, 256, 4, 2, 1),
						nn.BatchNorm2d(256),
						nn.ReLU()
						)
		self.layer3 = nn.Sequential(
						nn.ConvTranspose2d(256, 128, 4, 2, 1),
						nn.BatchNorm2d(128),
						nn.ReLU()
						)
		self.layer4 = nn.Sequential(
						nn.ConvTranspose2d(128, 1, 4, 2, 1),
						nn.Tanh()
						)

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		return x

disc = Discriminator()
gen = Generator()

criterion = nn.BCELoss()
optimizer_gen = optim.Adam(gen.parameters(), lr = 0.0001, betas = (0.5, 0.999))
optimizer_disc = optim.Adam(disc.parameters(), lr = 0.0001, betas = (0.5, 0.999))

def train_disc(real_imgs, fake_imgs, real_labels, fake_labels):
	disc.train()
	optimizer_disc.zero_grad()

	real_outputs = disc(real_imgs)
	real_loss = criterion(real_outputs, real_labels)

	fake_outputs = disc(fake_imgs)
	fake_loss = criterion(fake_outputs, fake_labels)

	loss = real_loss + fake_loss

	loss.backward()
	optimizer_disc.step()

	return loss

def train_gen(fake_inputs, real_labels):
	gen.train()
	optimizer_gen.zero_grad()

	outputs = disc(fake_inputs)

	gen_loss = criterion(outputs, real_labels)

	gen_loss.backward()
	optimizer_gen.step()

	return gen_loss

epochs = 20

for epoch in range(epochs):
	for i, data in enumerate(train_load):
		real_imgs, _ = data
		disc_noise = Variable(torch.randn(real_imgs.size(0), 100, 1, 1), requires_grad = False)

		fake_imgs = gen(disc_noise).detach()

		real_labels = Variable(torch.ones(real_imgs.size(0)), requires_grad = False)

		fake_labels = Variable(torch.zeros(real_imgs.size(0)), requires_grad = False)

		# if i%10 == 0:
		disc_loss = train_disc(Variable(real_imgs), fake_imgs, real_labels, fake_labels)

		gen_noise = Variable(torch.randn(real_imgs.size(0), 100, 1, 1), requires_grad = False)

		gen_img = gen(gen_noise)

		g_loss = train_gen(gen_img, real_labels)

		if i%100==0:
			print "Epoch : {} Step : {} Discriminator Loss : {} Generator Loss : {}".format(epoch, i, disc_loss.data[0], g_loss.data[0])
			display(gen_img)