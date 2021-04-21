"""
Created on Sat Nov 18 23:12:08 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import numpy as np

import torch
from torch.optim import Adam
from torchvision import models

from misc_functions import preprocess_image, recreate_image, save_image


class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        self.hook_layer()
        # Generate a random image
        random_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        # Process image and return variable
        processed_image = preprocess_image(random_image, False)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 31):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if index == self.selected_layer:
                    # (forward hook function triggered)
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(processed_image)
            # Save image
            if i % 5 == 0:
                im_path = '../generated/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)

    def visualise_layer_without_hooks(self):
        # Process image and return variable
        # Generate a random image
        random_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        # Process image and return variable
        processed_image = preprocess_image(random_image, False)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 31):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                x = layer(x)
                if index == self.selected_layer:
                    # Only need to forward until the selected layer is reached
                    # Now, x is the output of the selected layer
                    break
            # Here, we get the specific filter from the output of the convolution operation
            # x is a tensor of shape 1x512x28x28.(For layer 17)
            # So there are 512 unique filter outputs
            # Following line selects a filter from 512 filters so self.conv_output will become
            # a tensor of shape 28x28
            self.conv_output = x[0, self.selected_filter]
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(processed_image)
            # Save image
            if i % 5 == 0:
                im_path = '../generated/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)


if __name__ == '__main__':
    cnn_layer = 4
    filter_pos = 5
    # Fully connected layer is not needed
    pretrained_model = models.vgg16(pretrained=True).features
#     pretrained_model = models.resnet50(pretrained=True).layer2

    
    layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer, filter_pos)

    
#     print("models.vgg16(pretrained=True).features")
#     for index, layer in enumerate(models.vgg16(pretrained=True).features):
#         print("   ",index,layer)
#     print("models.resnet50(pretrained=True).layer1")
#     for index, layer in enumerate(models.resnet50(pretrained=True).layer1):
#         print("   ",index,layer)
#     print("models.resnet50(pretrained=True).layer2")
#     for index, layer in enumerate(models.resnet50(pretrained=True).layer2):
#         print("   ",index,layer)
#     print("models.resnet50(pretrained=True).layer3")
#     for index, layer in enumerate(models.resnet50(pretrained=True).layer3):
#         print("   ",index,layer)
#     print("models.resnet50(pretrained=True).layer4")
#     for index, layer in enumerate(models.resnet50(pretrained=True).layer4):
#         print("   ",index,layer)


    if 1:
        model = torch.nn.modules.container.Sequential(models.resnet50(pretrained=True))
        print(type(model))
        for index, layer in enumerate(model):
            print(index,layer)
        random_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        # Process image and return variable
        processed_image = preprocess_image(random_image, False)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 31):
            optimizer.zero_grad()
            x = processed_image
            for index, layer in enumerate(model):
                x = layer(x)
                if index == cnn_layer:
                    break
            conv_output = x[0, filter_pos]
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            created_image = recreate_image(processed_image)
            # Save image
            if i % 5 == 0:
                im_path = '../generated/layer_vis_l' + str(cnn_layer) + \
                    '_f' + str(filter_pos) + '_iter' + str(i) + '.jpg'
                save_image(created_image, im_path)

    
    
    # Layer visualization with pytorch hooks
    # layer_vis.visualise_layer_with_hooks()

    # Layer visualization without pytorch hooks
#     layer_vis.visualise_layer_without_hooks()
    
#     l1 = []
#     l2 = []
#     for m in dir(models.vgg16(pretrained=True)):
#         if m[0]!='_': l1.append(m)
#     for m in dir(models.resnet50(pretrained=True)):
#         if m[0]!='_': l2.append(m)
#     print(l1)
#     print(l2)
#     aim = "<class 'torch.nn.modules.container.Sequential'>"
#     for m in dir(models.vgg16(pretrained=True)):
#         if m[0]!='_':
#             if (aim==eval("str(type(models.vgg16(pretrained=True).{0}))".format(m))):
#                 print("vgg16",m,aim)
#     for m in dir(models.resnet50(pretrained=True)):
#         if m[0]!='_':
#             if (aim==eval("str(type(models.resnet50(pretrained=True).{0}))".format(m))):
#                 print("resnet50",m,aim)
# vgg16 classifier <class 'torch.nn.modules.container.Sequential'>
# vgg16 features <class 'torch.nn.modules.container.Sequential'>
# resnet50 layer1 <class 'torch.nn.modules.container.Sequential'>
# resnet50 layer2 <class 'torch.nn.modules.container.Sequential'>
# resnet50 layer3 <class 'torch.nn.modules.container.Sequential'>
# resnet50 layer4 <class 'torch.nn.modules.container.Sequential'>
    