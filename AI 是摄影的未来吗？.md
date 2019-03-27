# AI 是摄影的未来吗？
原文地址： https://medium.com/sfu-big-data/ai-the-future-of-photography-c7c80baf993b

-------
机器学习和人工生成图片将可能如何替代摄影。

### 导论
大部分人听到「AI」，「机器学习」或者「机器人」这些词都会脑补出一个像科幻电影里那样能说会走的安卓机器人，并马上默认时间在遥远的未来。

![1_YLUnptx-eVY5lhSs-tkS1](media/15535891376477/1_YLUnptx-eVY5lhSs-tkS1w.gif)
来源 : Giphy

抱歉了您嘞！AI 早已伴随我们多年，栖居于你的智能手机（Siri/ Google Assistant 万岁!）、你爱车的 GPS 系统、甚至当你读完这篇文章后研究下一篇给你推荐什么。然而在过去的几年，没有哪个领域比计算机视觉更受其影响。

随着技术的出现，超高分辨率的吸睛图像变得越来越普遍。 人们不再需要使用 Photoshop 和 CorelDRAW 等工具来增强和改变图像。 为了得到最佳图像，AI 已经被用于图像增强和操纵的每个方面。然而，最新的点子实际上是使用 AI 来生成合成图像。

你看过的几乎每一张图像都要么是被抓拍的照片，要么是被活生生的人手动创作的。可能有上百种手动生成图像的工具，但它们都确确实实需要一个人来主导这个过程。然而，试想一个计算机程序可以独立从头画出任何你让它画的东西。[微软的绘画机器人](https://drawingbot.azurewebsites.net/) 可能是第一个且唯一一个使这成为可能的技术。想象一下在不久的将来，你只需要下载一个 app 在你的智能手机，给它一点诸如「我要一幅我站在埃菲尔铁塔旁的图片」的指示。不就是图片吗，要啥有啥。

![](https://i.loli.net/2019/03/27/5c9b0e2ee3f7f.jpg)
来源: Blazepress

### 生成性对抗网络 (GANs，Generative Adversarial Networks)

> 「GANs 是机器学习领域近十年来最有意思的点子。」

> [— Yann LeCun](https://en.wikipedia.org/wiki/Yann_LeCun)

合成图像生成的基础是生成性对抗网络。自它们自 2014 年由 Ian Goodfellow 和他的团队发现和发布以来，GANs 一直是深度学习最迷人、最广泛使用的方面之一。这项技术，即对抗性训练的核心的无限应用，不仅包括计算机视觉领域，还包括数据分析，机器人技术和预测建模。

所以 GANs 有什么了不起？

生成性对抗网络属于一组生成模型。 这表示他们的工作是在完全自动化的过程中创建或「生成」新数据。

![](https://i.loli.net/2019/03/27/5c9b0e2e09248.jpg)
Goodfellow 论文中的生成图像 ([Source](https://arxiv.org/abs/1406.2661))

顾名思义，GAN 实际上由两个相互竞争的个体神经网络组成（以`_adversarial_`方式）。 一个称为`_generator_`（生成器）的神经网络生成由随机噪声创建的新数据实例，而另一个`_discriminator_`（判别器）则评估它们的真实性。 也就是说，判别器决定它所审核的每个数据实例是否属于实际训练数据集。

#### 一个简单的例子

假设你的任务是仿出一幅一位著名画家的画。很不幸，你不知道这位画家是谁，也从没见过他/她的画。而你的任务是伪造一幅画并将其作为原件之一展出在拍卖会上。 所以，你决定尝试一下。 你只需要一些颜料和画布，对吧？ 然而，拍卖商不希望有人卖乱画的东西，而且只要真货，所以他们聘请了一名侦探提前验货。 「幸运」的是，侦探自己有著名艺术家原画的样本，当你展示你的涂鸦时，他立刻就能识破。

![](https://i.loli.net/2019/03/27/5c9b0e2ca2a47.jpg)
来源: GitHub

他拒绝了你的画，你决定再试一试。但这一次，你有一些有用的提示，当他评估你的画时，侦探会说漏嘴画应该是什么样子。

现在你又来碰运气了，这次的画应该会好一点。但侦探仍然没被说服，再一次拒绝了你。于是你一次又一次地尝试，每一次都利用反馈来修正画作，让它越来越好。（我们假设侦探对你无数次回来没意见。）最终，在上千次尝试后，你终于能够拿出一幅接近完美的复制品。当侦探看他的样本画作时，侦探对你拿给他的画已经真假莫辨了。

#### GAN 如何一步步工作

将相同的思维过程应用于神经网络的组合，GAN 的训练包括以下步骤：

![](https://i.loli.net/2019/03/27/5c9b0e2c9dc40.jpg)
一个基础 GAN 框架 (Source : [Medium](https://medium.freecodecamp.org/an-intuitive-introduction-to-generative-adversarial-networks-gans-7a2264a81394))

1. 首先生成器接收一些随机噪声并将其传递给判别器。
2. 由于判别器已经可以访问真实图像的数据集，因此将它们与从生成器接收的图像进行比较并评估其真实性。
3. 由于初始图像只是随机噪声，因此它将被判为假的。
4. 生成器通过改变其参数来不断尝试运气，以便产生更好的图像。
5. 随着训练的进行，生成假图像的生成器和检测它们的判别器这两个网络都变得越来越聪明。
6. 最终，生成器设法创建与真实图像数据集中的图像无法区分的图像。 判别器不够聪明，无法判断给定的图像是真实的还是假的。
7. 此时，训练结束，生成的图像即我们的最终结果。

![](https://i.loli.net/2019/03/27/5c9b0e2b26d9d.jpg)
我们自己的生成汽车标志图像的 GAN

#### 是时候看点代码了

这是一个用 PyTorch 实现的基本生成性网络：

```
import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
os.makedirs('../../data/mnist', exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                            d_loss.item(), g_loss.item()))

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)
```

来源 : [eriklindernoren](https://github.com/eriklindernoren)

#### 优劣

正如所有技术，GANs 也有它们独特的优劣之处。此处大致地总结一下。
使用 GANs 有如下潜在优势：

- 并不总是需要标记的数据来训练。

- 比其他生成模型（如信念网络，belief network）更快地生成样本，因为它们不需要按顺序生成样本中的不同条目。

- 更容易训练依赖于对数配分函数梯度的蒙特卡罗近似的生成模型。 因为蒙特卡罗方法在高维空间中不能很好地工作，所以这种生成模型对于像 ImageNet 训练这样的现实任务来说效果不佳。

![](https://i.loli.net/2019/03/27/5c9b0e2b26a98.jpg)

- 不会引入任何决定性偏置。 像变分自动编码器这样的某些生成方法会引入决定性偏置，因为它们优化了对数似然的下界，而不是似然度本身，这会导致 VAEs 学习生成比 GANs 更模糊的样本。

同样的， GANs 也有如下不足：

- GANs 特别难训练。 这些网络试图优化的函数是一个基本上没有封闭形式的损失函数（不像标准损失函数，如对数损失或平方误差）。 因此，优化这种损失函数非常困难，并且需要对网络结构和训练协议进行大量的反复试错。

- 特别是对于图像生成，没有适当的措施来评估准确性。 一个合成图像在计算机看来是可通过的，实际结果却是非常主观的且取决于人类观察者。 而我们有初始得分（IS，Inception Score）和 FID（Fréchet Inception DistanceFrechet） 等函数来衡量它们的表现。

### GANs 的应用

最有意思的部分来了。我们能用 GANs 做好多有趣的事情。在它的所有潜在用途里，GANs 在计算机视觉领域可谓是广阔天地，大有可为。

#### 文本转图像

这个概念有几种实现，例如 TAC-GAN ——文本条件辅助分类器生成对抗网络，Text Conditioned Auxiliary Classifier Generative Adversarial Network。 它们用于根据文本描述合成图像。

![](media/15535891376477/15536654721779.jpg)
左: TAC-GAN 的结构 右: 向该网络输入一行文本后的结果

#### 域迁移

GAN在风格转移等概念中很受欢迎。 

它包括使用一种称为 CGAN（条件生成对抗网络，Conditional Generative Adversarial Networks）的特殊 GAN 进行图像到图像的转换。 绘画和概念设计从未如此简单。 然而，虽然 GAN 可以素描完成像下面这个钱包这样简单的绘图，但绘制更复杂的东西，如完美的人脸，目前还不是 GAN 的强项。 事实上，对于某些物体来说，它的结果简直是噩梦。


![](https://i.loli.net/2019/03/27/5c9b0e2cc9091.jpg)
CGAN 结果 [pix2pix ](https://github.com/phillipi/pix2pix)(来源:Github)

#### 图像补全和扩展(Inpainting and Outpainting)

图像补全和扩展是生成性网络的两个超赞应用。前者包括图像内的填充或噪声，有时候也被看作图像修复。 例如，给定具有孔或间隙的图像，GAN 应该能够以“可通过的”方式对其进行校正。 而扩展则涉及使用网络自己的学习来想象图像在其当前边界之外的样子。

![](https://i.loli.net/2019/03/27/5c9b0e2d1b227.jpg)
图像补全（左）和扩展（右）结果 [来源:Github]

#### 面部合成

有了生成性网络，生成一张多角度的面部图像变得可行。这就是为什么面部识别不需要你的几百个面部样本，而使用一个样本就足够。 不仅如此，生成人造面孔也不再是难事。 NVIDIA 最近使用他们的 GAN 2.0 使用 Celeba Hq 数据集生成高清分辨率的人造人脸，这是高分辨率合成图像生成的第一个例子。

![](https://i.loli.net/2019/03/27/5c9b0e2e9c250.jpg)
由渐进式 GAN 生成的假想名人面孔 (来源 : NVIDIA)

#### GANimation

复杂的小方法也变得可能，例如改变面部运动。 基于 PyTorch 的[GANimation](https://github.com/albertpumarola/GANimation) 将自己定义为「基于解剖学知识从单张图像生成人脸表情动画」。

![](media/15535891376477/15536655265472.jpg)
[GANimation 官方实现](http://www.albertpumarola.com/research/GANimation/index.html) (来源: [Github](https://github.com/albertpumarola/GANimation))

#### 绘画转照片

使用 GANs 使图像更逼真的另一个例子是轻松将（非常好的）绘画变成照片。 这是使用一种叫做 CycleGAN 的特殊 GAN 完成的，它使用两个生成器和两个判别器。 我们调用一个生成器`_G_`，并将它从`_X_`域转换为`_Y_`域。 另一个生成器名为`_F_`，并将图像从`_Y_`转换为`_X_`。 每个生成器都有一个相应的判别器，试图区分合成图像和真实图像。

![](https://i.loli.net/2019/03/27/5c9b0e2f790a3.jpg)
CycleGAN 结果。 (Source: [Github](https://github.com/junyanz/CycleGAN))

### 未来会怎样
在不远的将来，机器学习和 GANs 无疑将对影像界造成巨大影响。 目前，这项技术已经能够靠文本输入生成简单图像。 然而，在可预见的未来，它不仅能够创建高分辨率的精确图像，还能够创建整个视频。 想象一下只需要将剧本输入一个 GAN 就能生成整部电影会是怎样。 不仅如此，每个人都可以使用简单的交互式应用程序来创建自己的电影（甚至可以自己主演！）。 这项技术会成为真的摄影，导演和表演的终点吗？

令人惊叹的技术也意味着被用于恶意目的的潜在性。 完美假图像还需要一种识别和检测它们的方法。 这样的图像生成亟需规定管理。目前，GANs 已经被用于制作虚假视频或「Deepfakes」，这些视频以负面方式使用，例如生成虚假的名人色情视频或让人们在他们不知情的情况下说话。 将技术用于合成大众可用的音频和视频生成的后果是很可怕的。

人工生成图像是把双刃剑，特别是当大众对它知之甚少的时候。生成性对抗网络是一个非常有用和危险的工具。它无疑将重塑技术世界，但对于如何重塑，我们只能深思。