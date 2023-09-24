import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import torch

class Plot2D():
    def Contour2D(nodecoords, sol, savefig=False, figname=''):
        fig, ax = plt.subplots(constrained_layout=True)
        x, y=nodecoords[:, 1],nodecoords[:, 2]
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        X, Y = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
        Z = griddata(nodecoords, sol, (X, Y), method='linear') # cubic, linear, nearest
        cs= plt.contourf(X, Y, Z, 100, cmap='jet', levels=200)
        fig.colorbar(cs)
        ax.axis('equal')

        if savefig:
            if len(figname)>4:
                fig.savefig(figname,dpi=300,bbox_inches='tight')
                print('save results to ',figname)
            else:
                fig.savefig('result.jpg',dpi=300,bbox_inches='tight')
                print('save result to result.jpg')


class StrgridEngine:
    def __init__(self, dimension=2, grid_size=(101, 101)):
        self.dimension = dimension
        self.grid_size = grid_size

    def generate_structure_grid(self):
        # Generate a structured grid of points
        x_points = torch.linspace(0, 1, self.grid_size[0])
        y_points = torch.linspace(0, 1, self.grid_size[1])

        # Create grid using meshgrid
        grid_x, grid_y = torch.meshgrid(x_points, y_points)

        # Flatten the grid
        flattened_grid = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)

        return flattened_grid
