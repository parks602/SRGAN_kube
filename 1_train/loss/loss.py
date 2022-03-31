import torch
from torch import nn


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, out_labels, out_images, target_images, fake_landsea, fake_hight, real_landsea, real_hight):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        #perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        landsea_loss = self.mse_loss(fake_landsea, real_landsea)
        hight_loss = self.mse_loss(fake_hight, real_hight)
        image_los  = image_loss * 0.9 + landsea_loss *0.5 + hight_loss*0.00002
        # TV Loss
        return image_los +  adversarial_loss, image_loss*0.9, landsea_loss*0.2, hight_loss*0.00002, adversarial_loss


    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

