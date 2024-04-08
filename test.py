import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch
from matplotlib import colors
import matplotlib.gridspec as gridspec

def OpticalFlow3(deformation):
    deformation_flow = deformation.squeeze(0).cpu().numpy()

    deformation_flow = deformation_flow[::8, ::8, :]
    # Create a grid of coordinates for each pixel
    x = np.arange(0, 8, 1)
    y = np.arange(0, 8, 1)
    x, y = np.meshgrid(x, y)

    # Extract the flow vectors in the x and y directions
    u = deformation_flow[:, :, 0]
    v = deformation_flow[:, :, 1]

    fig, ax = plt.subplots()
    image = np.ones((8, 8, 3))
    # ax.set_facecolor('white')
    ax.imshow(image)
    # Create the arrow plot
    plt.quiver(x, y, u, v, color='green', scale=2, scale_units ='xy')  # Plot the flow vectors as arrows
    plt.axis('off')
    plt.savefig('D:/Project/FOMM_rong/test-vis/arrow_plot.png',bbox_inches='tight',pad_inches=0.0)  # Save the arrow plot as an image file