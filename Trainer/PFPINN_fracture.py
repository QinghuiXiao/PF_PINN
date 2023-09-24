import matplotlib.pyplot as plt
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from Common import NeuralNet
import numpy as np

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(128)

class Pinns:
    def __init__(self, n_int_, n_sb_, n_tb_, nx_, ny_, save_dir_, pre_model_save_path_, device_):
        self.pre_model_save_path = pre_model_save_path_
        self.save_dir = save_dir_

        self.n_int = n_int_
        self.n_sb = n_sb_
        self.n_tb = n_tb_
        self.nx = nx_
        self.ny = ny_

        #parameters
        self.E = 210
        self.nu = 0.3
        self.G = (self.E / (2 * (1 + self.nu)))
        self.g = 0.0027
        self.l = 0.01

        #self.U_0 = 0
        self.device = device_
        # Extrema of the solution domain (t,x,y) in [0,1]x[0,1]x[0,1]
        self.domain_extrema = torch.tensor([[0, 1],  # Time dimension
                                            [0, 1],  # x dimension
                                            [0, 1]])  # y dimension
        # Number of space dimensions
        self.space_dimensions = 2

        # F Dense NN to approximate the solution of the underlying heat equation
        self.approximate_solution = NeuralNet(input_dimension=3, output_dimension=3,
                                              n_hidden_layers=4,
                                              neurons=20,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=42)
        '''self.approximate_solution = MultiVariatePoly(self.domain_extrema.shape[0], 3)'''
        if pre_model_save_path_:
            self.load_checkpoint()

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        # Training sets S_sb, S_tb, S_int as torch dataloader
        self.training_set_sb, self.training_set_tb, self.training_set_int = self.assemble_datasets()

    ################################################################################################
    # Function to linearly transform a tensor whose value are between 0 and 1
    # to a tensor whose values are between the domain extrema
    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    ################################################################################################
    # Function returning the input-output tensor required to assemble the training set S_tb corresponding to the temporal boundary
    def add_temporal_boundary_points(self):
        # 从时间坐标中获取初始时间
        t_initial = torch.tensor([self.domain_extrema[0, 0]])
        input_tb_time = torch.tile(t_initial[:, None], [self.n_tb, 1])

        x_01 = torch.linspace(0, 0.5, 51)
        y_01 = torch.cat([torch.linspace(0, 0.49, 50), torch.linspace(0.51, 1, 50)])
        grid_x01, grid_y01 = torch.meshgrid(x_01, y_01)
        input_tb_01 = torch.stack((grid_x01.reshape(-1), grid_y01.reshape(-1)), dim=1)

        x_02 = torch.linspace(0.51, 1, 50)
        y_02 = torch.linspace(0, 1, 101)
        grid_x02, grid_y02 = torch.meshgrid(x_02, y_02)
        input_tb_02 = torch.stack((grid_x02.reshape(-1), grid_y02.reshape(-1)), dim=1)
        input_tb_0 = torch.cat([input_tb_01, input_tb_02], 0)
        print(input_tb_0.shape)
        print(input_tb_0)
        # 在空间坐标上均匀取样裂纹部分
        x_1 = torch.linspace(0, 0.5, 51)
        y_1 = torch.full((51, 1), 0.5)
        input_tb_1 = torch.cat([x_1.unsqueeze(1), y_1], dim=1)
        print(input_tb_1.shape)
        print(input_tb_1)
        input_tb = torch.cat([input_tb_0, input_tb_1], 0)
        input_tb = torch.cat((input_tb, input_tb_time), dim=1)
        input_tb = torch.tensor(input_tb, dtype=torch.float32)
        print(input_tb)
        return input_tb

    # Function returning the input-output tensor required to assemble the training set S_sb corresponding to the spatial boundary
    def add_spatial_boundary_points(self):
        x0 = self.domain_extrema[1, 0]
        xL = self.domain_extrema[1, 1]
        y0 = self.domain_extrema[2, 0]
        yL = self.domain_extrema[2, 1]

        input_sb = self.convert(self.soboleng.draw(self.n_sb))

        input_sb_L = torch.clone(input_sb)
        input_sb_L[:, 1] = x0
        input_sb_D = torch.clone(input_sb)
        input_sb_D[:, 2] = y0
        input_sb_R = torch.clone(input_sb)
        input_sb_R[:, 1] = xL
        input_sb_U = torch.clone(input_sb)
        input_sb_U[:, 2] = yL
        input_sb = torch.cat([input_sb_U, input_sb_D, input_sb_L, input_sb_R], 0)
        print(input_sb.shape)
        print(input_sb)
        return input_sb

    def add_interior_points(self):
        t = torch.linspace(self.domain_extrema[0, 0], self.domain_extrema[0, 1], 5)
        input_int_time = torch.tile(t[:, None], [self.nx*self.ny, 1])
        # 在空间坐标上均匀取样，使用 linspace 函数
        x_samples = torch.linspace(self.domain_extrema[1, 0], self.domain_extrema[1, 1], self.nx)
        y_samples = torch.linspace(self.domain_extrema[2, 0], self.domain_extrema[2, 1], self.ny)
        # 创建网格坐标
        grid_x, grid_y = torch.meshgrid(x_samples, y_samples)
        input_int_space = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)
        input_int_space = torch.tile(input_int_space, [5, 1])
        # 将时间坐标与空间坐标连接在一起
        input_int = torch.cat((input_int_space, input_int_time), dim=1)

        return input_int

    def assemble_datasets(self):
        input_sb = self.add_spatial_boundary_points()  # S_sb
        input_tb = self.add_temporal_boundary_points()  # S_tb
        input_int = self.add_interior_points()  # S_int

        training_set_sb = DataLoader(torch.utils.data.TensorDataset(input_sb),
                                     batch_size=2 * self.space_dimensions * self.n_sb, shuffle=False)

        training_set_tb = DataLoader(torch.utils.data.TensorDataset(input_tb),
                                     batch_size=self.n_tb, shuffle=False)

        training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int),
                                      batch_size=5 * self.n_int, shuffle=False)

        return training_set_sb, training_set_tb, training_set_int

    def compute_ic0_residual(self, input_tb):

        u = self.approximate_solution(input_tb)
        u1 = u[:, 0].reshape(-1, 1)
        u2 = u[:, 1].reshape(-1, 1)
        c = u[:, 2].reshape(-1, 1)

        residual_u1 = u1
        residual_u2 = u2
        residual_c0 = c

        return residual_u1.reshape(-1, ), residual_u2.reshape(-1, ), residual_c0.reshape(-1, )

    def compute_ic1_residual(self, input_tb):

        u = self.approximate_solution(input_tb)
        u1 = u[:, 0].reshape(-1, 1)
        u2 = u[:, 1].reshape(-1, 1)
        c = u[:, 2].reshape(-1, 1)

        residual_u1 = u1
        residual_u2 = u2
        residual_c1 = c - 1

        return residual_u1.reshape(-1, ), residual_u2.reshape(-1, ), residual_c1.reshape(-1, )

    def compute_bc_residual(self, input_sb):

        u = self.approximate_solution(input_sb)
        u1 = u[:, 0].reshape(-1, 1)
        u2 = u[:, 1].reshape(-1, 1)
        #c = u[:, 2].reshape(-1, 1)

        residual_u1 = u1
        residual_u2 = u2
        #residual_c = c

        return residual_u1.reshape(-1, ), residual_u2.reshape(-1, )
    # Function to compute the PDE residuals
    def compute_pde_residual(self, input_int):

        u = self.approximate_solution(input_int)
        u1 = u[:, 0].reshape(-1, 1)
        u2 = u[:, 1].reshape(-1, 1)
        c = u[:, 2].reshape(-1, 1)

        grad_u1 = torch.autograd.grad(u1.sum(), input_int, create_graph=True)[0]
        u1_t, u1_1, u1_2 = grad_u1[:, 0], grad_u1[:, 1], grad_u1[:, 2]

        grad_u1_1 = torch.autograd.grad(u1_1.sum(), input_int, create_graph=True)[0]
        u1_11, u1_12 = grad_u1_1[:, 1], grad_u1_1[:, 2]
        grad_u1_2 = torch.autograd.grad(u1_2.sum(), input_int, create_graph=True)[0]
        u1_21, u1_22 = grad_u1_2[:, 1], grad_u1_2[:, 2]

        grad_u2 = torch.autograd.grad(u2.sum(), input_int, create_graph=True)[0]
        u2_t, u2_1, u2_2 = grad_u2[:, 0], grad_u2[:, 1], grad_u2[:, 2]

        grad_u2_1 = torch.autograd.grad(u2_1.sum(), input_int, create_graph=True)[0]
        u2_11, u2_12 = grad_u2_1[:, 1], grad_u2_1[:, 2]
        grad_u2_2 = torch.autograd.grad(u2_2.sum(), input_int, create_graph=True)[0]
        u2_21, u2_22 = grad_u2_2[:, 1], grad_u2_2[:, 2]

        grad_c= torch.autograd.grad(c.sum(), input_int, create_graph=True)[0]
        c_t, c_1, c_2 = grad_c[:, 0], grad_c[:, 1], grad_c[:, 2]
        grad_c_1 = torch.autograd.grad(c_1.sum(), input_int, create_graph=True)[0]
        c_11 = grad_c_1[:, 1]
        grad_c_2 = torch.autograd.grad(c_2.sum(), input_int, create_graph=True)[0]
        c_22 = grad_c_2[:, 2]

        sigma_11 = ((2 * self.nu * self.G / (1 - self.nu)) * (u1_1 + u2_2) + (2 * self.G * u1_1)) * (1 - c) ** 2
        sigma_22 = ((2 * self.nu * self.G / (1 - self.nu)) * (u1_1 + u2_2) + (2 * self.G * u2_2)) * (1 - c) ** 2
        sigma_12 = self.G * (u1_2 + u2_1) * (1 - c) ** 2

        W = 0.5 * (sigma_11 * u1_1 + sigma_22 * u2_2 + 2 * sigma_12 * (u1_2 + u2_1))

        residual_PDE_1 = ((2 / (1 - self.nu)) * u1_11 + u1_22 + ((1 + self.nu) / (1 - self.nu)) * u2_21) * (
                    (1 - c) ** 2) - 4 * (1 - c) * c_1 * ((self.nu / (1 - self.nu) * (u1_1 + u2_2)) + u1_1) - 2 * (
                            1 - c) * c_2 * (u1_2 + u2_1)
        residual_PDE_2 = ((2 / (1 - self.nu)) * u2_22 + u2_11 + ((1 + self.nu) / (1 - self.nu)) * u1_12) * (
                    (1 - c) ** 2) - 4 * (1 - c) * c_2 * ((self.nu / (1 - self.nu) * (u1_1 + u2_2)) + u2_2) - 2 * (
                            1 - c) * c_1 * (u1_2 + u2_1)
        residual_PDE_3 = c * self.g / self.l - 2 * (1 - c) * W - self.g * self.l * (c_11 + c_22)

        return residual_PDE_1.reshape(-1, ), residual_PDE_2.reshape(-1, ), residual_PDE_3.reshape(-1, )

    # Function to compute the total loss (weighted sum of spatial boundary loss, temporal boundary loss and interior loss)
    def compute_loss(self, inp_train_sb,  inp_train_tb,  inp_train_int, verbose=True):

        r_int_1, r_int_2, r_int_3 = self.compute_pde_residual(inp_train_int)
        r_sb_u1, r_sb_u2 = self.compute_bc_residual(inp_train_sb)
        r_tb_u1_0, r_tb_u2_0, r_tb_c_0, = self.compute_ic0_residual(inp_train_tb[0:10150, :])
        r_tb_u1_1, r_tb_u2_1, r_tb_c_1, = self.compute_ic1_residual(inp_train_tb[10150:, :])

        loss_sb = torch.mean(abs(r_sb_u1) ** 2) + torch.mean(abs(r_sb_u2) ** 2)
        loss_tb = torch.mean(abs(r_tb_u1_0) ** 2) + torch.mean(abs(r_tb_u2_0) ** 2) + torch.mean(
            abs(r_tb_c_0) ** 2) + torch.mean(abs(r_tb_u1_1) ** 2) + torch.mean(abs(r_tb_u2_1) ** 2) + torch.mean(
            abs(r_tb_c_1) ** 2)
        loss_int = torch.mean(abs(r_int_1) ** 2) + torch.mean(abs(r_int_2) ** 2) + torch.mean(abs(r_int_3) ** 2)

        loss_u = loss_sb + loss_tb

        loss = torch.log10(loss_sb + loss_tb + loss_int)
        if verbose: print("Total loss: ", round(loss.item(), 4), "| PDE Loss: ", round(torch.log10(loss_u).item(), 4),
                          "| Function Loss: ", round(torch.log10(loss_int).item(), 4))

        return loss

    ################################################################################################
    def fit(self, num_epochs, optimizer, verbose=True):
        history = list()

        # Loop over epochs
        for epoch in range(num_epochs):
            if verbose: print("################################ ", epoch, " ################################")

            for j, (inp_train_sb, inp_train_tb, inp_train_int) in enumerate(
                    zip(self.training_set_sb, self.training_set_tb, self.training_set_int)):
                def closure():
                    optimizer.zero_grad()
                    loss = self.compute_loss(inp_train_sb, inp_train_tb, inp_train_int,
                                             verbose=verbose)
                    loss.backward()

                    history.append(loss.item())
                    return loss

                optimizer.step(closure=closure)

        print('Final Loss: ', history[-1])
        return history

    def save_checkpoint(self):
        '''save model and optimizer'''
        torch.save({
            'model_state_dict': self.approximate_solution.state_dict()
        }, self.save_dir)

    def load_checkpoint(self):
        '''load model and optimizer'''
        checkpoint = torch.load(self.pre_model_save_path)
        self.approximate_solution.load_state_dict(checkpoint['model_state_dict'])
        print('Pretrained model loaded!')

n_int = 10201
n_sb = 1000
n_tb = 10201
nx = 101
ny = 101
#pre_model_save_path = './results/LBFGS_sqloss_test.pt'
pre_model_save_path = None
save_path = './results/LBFGS_sqloss_test.pt'

device = torch.device('cuda:0')

pinn = Pinns(n_int, n_sb, n_tb, nx, ny, save_path, pre_model_save_path, device)
pinn.approximate_solution.to(device)

n_epochs = 1
optimizer_LBFGS = optim.LBFGS(pinn.approximate_solution.parameters(),
                              lr=float(0.5),
                              max_iter=50000,
                              max_eval=50000,
                              history_size=150,
                              line_search_fn="strong_wolfe",
                              tolerance_change=1.0 * np.finfo(float).eps)
optimizer_ADAM = optim.Adam(pinn.approximate_solution.parameters(), lr=float(0.001), weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ADAM, step_size= 50, gamma= 0.95)
optimizer = optimizer_LBFGS

if pre_model_save_path:
    pinn.load_checkpoint()

hist = pinn.fit(num_epochs=n_epochs,
                optimizer=optimizer,
                verbose=True)
pinn.save_checkpoint()

#pinn.plotting()

# Plot the input training points
input_sb_, output_sb_ = pinn.add_spatial_boundary_points()
input_tb_, output_tb_ = pinn.add_temporal_boundary_points()
input_int_, output_int_ = pinn.add_interior_points()

plt.figure(figsize=(5, 5), dpi=150)
plt.scatter(input_sb_[:, 1].detach().numpy(), input_sb_[:, 2].detach().numpy(), c='blue', label="Boundary Points")
plt.scatter(input_int_[:, 1].detach().numpy(), input_int_[:, 2].detach().numpy(), c='green', label="Interior Points")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

print(input_tb_[0:n_tb, :])
print(input_tb_[n_tb:, :])

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

# Rest of the code remains the same...

# Create a 3D figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for spatial boundary points
ax.scatter(input_sb_[:, 1].detach().numpy(), input_sb_[:, 2].detach().numpy(), input_sb_[:, 0].detach().numpy(), c='blue', label="Boundary Points")

# Scatter plot for interior points
ax.scatter(input_int_[:, 1].detach().numpy(), input_int_[:, 2].detach().numpy(), input_int_[:, 0].detach().numpy(), c='orange', label="Interior Points")

# Scatter plot for initial points
ax.scatter(input_tb_[:, 1].detach().numpy(), input_tb_[:, 2].detach().numpy(), input_tb_[:, 0].detach().numpy(), c='green', label="Initial Points")

# Set labels for each axis
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('t')

# Add a legend
ax.legend()

# Show the 3D plot
plt.show()