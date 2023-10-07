import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from .model import NeuralNet
from tqdm import tqdm
from copy import deepcopy
import csv
from .utils import StrgridEngine

#Continuous time model
class Pinns:
    def __init__(self, config):
        self.config = config
        self.pre_model_save_path = config.pre_model_save_path
        self.save_dir = config.save_path
        self.n_int = config.n_int
        self.n_sb = config.n_sb
        self.n_tb = config.n_tb
        self.nx = config.nx
        self.ny = config.ny
        self.optimizer_name = config.optimizer
        self.device = config.device
        #parameters
        self.E = 210
        self.nu = 0.3
        self.G = (self.E / (2 * (1 + self.nu)))
        self.g = 0.0027
        self.l = 0.1

        # Extrema of the solution domain (t,x,y) in [0,1]x[0,1]x[0,1]
        self.domain_extrema = torch.tensor([[0, 0.1],  # Time dimension
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
                                              retrain_seed=42).to(self.device)
        '''self.approximate_solution = MultiVariatePoly(self.domain_extrema.shape[0], 3)'''
        if self.pre_model_save_path:
            self.load_checkpoint()

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        self.strueng = StrgridEngine(dimension=2, grid_size=(self.nx, self.ny))

        # Training sets S_sb, S_tb, S_int as torch dataloader
        self.training_set_sb, self.training_set_tb, self.training_set_int = self.assemble_datasets()

        # Optimizer
        self.init_optimizer()

    ################################################################################################
    # Function to linearly transform a tensor whose value are between 0 and 1
    # to a tensor whose values are between the domain extrema
    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    def add_temporal_boundary_points(self):
        # 从时间坐标中获取初始时间
        t_initial = torch.tensor([self.domain_extrema[0, 0]])
        input_tb_time = torch.tile(t_initial[:, None], [self.n_tb, 1])

        x_01 = torch.linspace(0, 0.5, 26)
        y_01 = torch.cat([torch.linspace(0, 0.49, 25), torch.linspace(0.51, 1, 25)])
        grid_x01, grid_y01 = torch.meshgrid(x_01, y_01)
        input_tb_01 = torch.stack((grid_x01.reshape(-1), grid_y01.reshape(-1)), dim=1)

        x_02 = torch.linspace(0.51, 1, 25)
        y_02 = torch.linspace(0, 1, 51)
        grid_x02, grid_y02 = torch.meshgrid(x_02, y_02)
        input_tb_02 = torch.stack((grid_x02.reshape(-1), grid_y02.reshape(-1)), dim=1)
        input_tb_0 = torch.cat([input_tb_01, input_tb_02], 0)

        # 在空间坐标上均匀取样裂纹部分
        x_1 = torch.linspace(0, 0.5, 51)
        y_1 = torch.full((51, 1), 0.5)
        input_tb_1 = torch.cat([x_1.unsqueeze(1), y_1], dim=1)

        input_tb = torch.cat([input_tb_0, input_tb_1], 0)
        input_tb = torch.cat((input_tb, input_tb_time), dim=1)

        return input_tb

    # Function returning the input-output tensor required to assemble the training set S_sb corresponding to the spatial boundary
    def add_spatial_boundary_points(self):
        y0 = self.domain_extrema[2, 0]
        yL = self.domain_extrema[2, 1]

        input_sb = self.convert(self.soboleng.draw(self.n_sb))
        input_sb_D = torch.clone(input_sb)
        input_sb_D[:, 2] = y0
        input_sb_U = torch.clone(input_sb)
        input_sb_U[:, 2] = yL
        input_sb = torch.cat([input_sb_U, input_sb_D], 0)

        return input_sb

    def add_interior_points(self):
        t = torch.linspace(self.domain_extrema[0, 0], self.domain_extrema[0, 1], 5)
        input_int_time = torch.tile(t[:, None], [self.n_int, 1])
        input_int_space = self.strueng.generate_structure_grid()
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
        input_tb.requires_grad = True
        u = self.approximate_solution(input_tb)
        u1 = u[:, 0].reshape(-1, 1)
        u2 = u[:, 1].reshape(-1, 1)
        c = u[:, 2].reshape(-1, 1)

        residual_u1 = u1
        residual_u2 = u2
        residual_c0 = c

        return residual_u1.reshape(-1, ), residual_u2.reshape(-1, ), residual_c0.reshape(-1, )

    def compute_ic1_residual(self, input_tb):
        input_tb.requires_grad = True
        u = self.approximate_solution(input_tb)
        u1 = u[:, 0].reshape(-1, 1)
        u2 = u[:, 1].reshape(-1, 1)
        c = u[:, 2].reshape(-1, 1)

        residual_u1 = u1
        residual_u2 = u2
        residual_c1 = c - 1

        return residual_u1.reshape(-1, ), residual_u2.reshape(-1, ), residual_c1.reshape(-1, )

    def compute_bcU_residual(self, input_sb):
        input_sb.requires_grad = True
        u = self.approximate_solution(input_sb)
        u2 = u[:, 1].reshape(-1, 1)
        residual_U_2 = u2 - 0.001 * input_sb[:, 0]
        return residual_U_2.reshape(-1, )

    def compute_bcD_residual(self, input_sb):
        input_sb.requires_grad = True
        u = self.approximate_solution(input_sb)
        u1 = u[:, 0].reshape(-1, 1)
        u2 = u[:, 1].reshape(-1, 1)
        residual_D_1 = u1
        residual_D_2 = u2
        return residual_D_1.reshape(-1, ), residual_D_2.reshape(-1, )

    # Function to compute the PDE residuals
    def compute_pde_residual(self, input_int):
        input_int.requires_grad = True
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

        grad_c = torch.autograd.grad(c.sum(), input_int, create_graph=True)[0]
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

        residual_c = c_t

        return residual_PDE_1.reshape(-1, ), residual_PDE_2.reshape(-1, ), residual_PDE_3.reshape(-1, ), residual_c.reshape(-1, )

    # Function to compute the total loss (weighted sum of spatial boundary loss, temporal boundary loss and interior loss)
    def compute_loss(self, inp_train_sb,  inp_train_tb,  inp_train_int, verbose=True):

        r_int_1, r_int_2, r_int_3, r_int_c = self.compute_pde_residual(inp_train_int)
        r_sbU_u2 = self.compute_bcU_residual(inp_train_sb[0:self.n_sb, :])
        r_sbD_u1, r_sbD_u2 = self.compute_bcD_residual(inp_train_sb[self.n_sb:, :])
        r_tb_u1_0, r_tb_u2_0, r_tb_c_0, = self.compute_ic0_residual(inp_train_tb[0:430, :])
        r_tb_u1_1, r_tb_u2_1, r_tb_c_1, = self.compute_ic1_residual(inp_train_tb[430:, :])

        loss_sb = torch.mean(abs(r_sbU_u2) ** 2) + torch.mean(abs(r_sbD_u2) ** 2) + torch.mean(abs(r_sbD_u1) ** 2)
        loss_tb = torch.mean(abs(r_tb_u1_0) ** 2) + torch.mean(abs(r_tb_u2_0) ** 2) + torch.mean(
            abs(r_tb_c_0) ** 2) + torch.mean(abs(r_tb_u1_1) ** 2) + torch.mean(abs(r_tb_u2_1) ** 2) + torch.mean(
            abs(r_tb_c_1) ** 2)

        loss_int = torch.mean(abs(r_int_1) ** 2) + torch.mean(abs(r_int_2) ** 2) + torch.mean(abs(r_int_3) ** 2)
        loss_inequality = torch.mean(torch.relu(-r_int_c))
        loss = loss_sb + loss_tb + loss_int + loss_inequality * 10

        if verbose: print("Total loss: ", round(torch.log10(loss).item(), 4),
                          "| Inequality Loss: ", round(torch.log10(loss_inequality).item(), 4),
                          "| BC Loss: ", round(torch.log10(loss_sb).item(), 4),
                          "| IC Loss: ", round(torch.log10(loss_tb).item(), 4),
                          "| PDE Loss: ", round(torch.log10(loss_int).item(), 4))

        return loss, loss_sb, loss_tb, loss_int, loss_inequality

    def init_optimizer(self):
        '''Initialize optimizer'''
        if self.optimizer_name == "lbfgs":
            self.optimizer = torch.optim.LBFGS(self.approximate_solution.parameters(), lr=float(0.5),
                                               max_iter=self.config.max_iter,
                                               max_eval=50000, tolerance_change=1.0 * np.finfo(float).eps,
                                               history_size=150, line_search_fn="strong_wolfe")
        elif self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.approximate_solution.parameters(), lr=self.config.lr)
        else:
            raise NotImplementedError(f"Optimizer {self.optimizer_name} not implemented")

        # init scheduler
        if self.config.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                             step_size=self.config.scheduler_step_size,
                                                             gamma=self.config.scheduler_gamma)
            if self.config.optimizer == "lbfgs":
                raise NotImplementedError(f"Scheduler not implemented for optimizer {self.config.optimizer}")
        else:
            self.scheduler = None

    def fit(self, num_epochs, max_iter, lr, verbose=True):
        '''Train process'''
        # Training preparation
        self.approximate_solution.train()
        best_loss, best_epoch, best_state = np.inf, -1, None
        losses = []
        losses_sb = []
        losses_tb = []
        losses_int = []
        losses_inequality = []

        epoch = {
            "lbfgs": max_iter,
            "adam": num_epochs
        }[self.optimizer_name]

        # pbar = tqdm(range(epoch), desc = 'Epoch', colour='blue')
        def train_batch(batch_sb, batch_tb, batch_int):
            inp_train_sb = batch_sb[0].to(self.device)
            inp_train_tb = batch_tb[0].to(self.device)
            inp_train_int = batch_int[0].to(self.device)

            def closure():
                self.optimizer.zero_grad()
                loss, loss_sb, loss_tb, loss_int, loss_inequality = self.compute_loss(
                    inp_train_sb, inp_train_tb, inp_train_int, verbose=verbose)
                # backpropragation
                loss.backward()
                # recording
                losses.append(loss.item())
                losses_sb.append(loss_sb.item())
                losses_tb.append(loss_sb.item())
                losses_int.append(loss_int.item())
                losses_inequality.append(loss_inequality.item())

                if self.config.optimizer == "lbfgs":
                    pbar.set_postfix(loss=losses[-1])
                    pbar.update(1)
                return loss
            return closure

        # training
        if self.optimizer_name == "lbfgs":
            pbar = tqdm(total=self.config.max_iter, desc='Batch',
                        colour='blue')  # Progress bar for LBFGS based on batches
            # optimizer = torch.optim.LBFGS(self.approximate_solution.parameters(), lr=lr, max_iter=max_iter, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)

            for j, (batch_sb, batch_tb, batch_int) in enumerate(zip(
                    self.training_set_sb, self.training_set_tb, self.training_set_int)):
                self.optimizer.step(closure=train_batch(batch_sb, batch_tb, batch_int))
            pbar.close()
            self.save_checkpoint()

        elif self.optimizer_name == "adam":
            pbar = tqdm(total=num_epochs, desc='Epoch', colour='blue')  # Progress bar for Adam based on epochs
            # optimizer = torch.optim.Adam(self.approximate_solution.parameters(), lr=lr)
            for ep in range(num_epochs):
                for j, (batch_sb, batch_tb, batch_int) in enumerate(zip(
                        self.training_set_sb, self.training_set_tb, self.training_set_int)):

                    train_batch(batch_sb, batch_tb, batch_int)()
                    self.optimizer.step()
                    if self.config.use_scheduler:
                        self.scheduler.step()

                    # save model
                    if losses[-1] < best_loss:
                        best_epoch = ep
                        best_loss = losses[-1]
                        best_state = deepcopy(self.approximate_solution.state_dict())
                        best_loss = losses[-1]
                pbar.set_postfix(loss=losses[-1])
                pbar.update(1)
            pbar.close()

            self.approximate_solution.load_state_dict(best_state)
            self.save_checkpoint()

        with torch.no_grad():
            u_end = self.approximate_solution(list(self.training_set_tb)[0][0].to(self.device))
            solution = torch.cat((list(self.training_set_tb)[0][0], u_end), 1)
            np.savetxt('./result_0.01.txt', solution.cpu().numpy(), delimiter=',', header='t,x,y,u1,u2,c')
            output_filename = f"output_0.01.csv"
            with open(output_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(solution.cpu().numpy())

        # plot losses vs epoch
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(np.arange(len(losses)), losses, label="loss")
        ax.plot(np.arange(len(losses_sb)), losses_sb, label="loss_sb")
        ax.plot(np.arange(len(losses_tb)), losses_sb, label="loss_tb")
        ax.plot(np.arange(len(losses_int)), losses_int, label="loss_int")
        ax.plot(np.arange(len(losses_inequality)), losses_inequality, label="loss_inequality")
        if best_epoch != -1:
            ax.scatter([best_epoch], [best_loss], c='r', marker='o', label="best loss")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss vs Epoch')
        ax.legend()
        ax.set_xlim(left=0)
        ax.set_yscale('log')
        plt.savefig(f'loss.png')

        return losses

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

    def plotting(self):
        t = torch.linspace(0, 1, 11)
        x = torch.linspace(0, 1, 101)
        y = torch.linspace(0, 1, 101)
        grid = torch.meshgrid(t, x, y, indexing='ij')
        inputs = torch.stack(grid, dim=-1).reshape(-1, 3)

        outputs = self.approximate_solution(inputs).detach().numpy()

        u1 = outputs[:, 0]
        u2 = outputs[:, 1]
        c = outputs[:, 2]

        for i, t_val in enumerate(t):
            if i % 1 == 0:
                xx, yy = np.meshgrid(x, y, indexing='ij')
                u1_reshaped = u1[i * xx.size:(i + 1) * xx.size].reshape(xx.shape)
                u2_reshaped = u2[i * xx.size:(i + 1) * xx.size].reshape(xx.shape)
                c_reshaped = c[i * xx.size:(i + 1) * xx.size].reshape(xx.shape)
                print(f"u1 size: {u1_reshaped.size}, xx size: {xx.size}")

                fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
                im1 = axs[0].imshow(u1_reshaped, cmap='jet', origin='lower',
                                    extent=[x.min(), x.max(), y.min(), y.max()])
                axs[0].set_xlabel("x")
                axs[0].set_ylabel("y")
                plt.colorbar(im1, ax=axs[0])
                axs[0].grid(True, which="both", ls=":")
                im2 = axs[1].imshow(u2_reshaped, cmap='jet', origin='lower',
                                    extent=[x.min(), x.max(), y.min(), y.max()])
                axs[1].set_xlabel("x")
                axs[1].set_ylabel("y")
                plt.colorbar(im2, ax=axs[1])
                axs[1].grid(True, which="both", ls=":")
                im3 = axs[2].imshow(c_reshaped, cmap='jet', origin='lower', extent=[x.min(), x.max(), y.min(), y.max()])
                axs[2].set_xlabel("x")
                axs[2].set_ylabel("y")
                plt.colorbar(im3, ax=axs[2])
                axs[2].grid(True, which="both", ls=":")
                axs[0].set_title(f"u1 at t = {t_val}")
                axs[1].set_title(f"u2 at t = {t_val}")
                axs[2].set_title(f"c at t = {t_val}")
                plt.savefig(f"./plotting_t_{t_val}.png", dpi=150)

#Discrete time model
class Pinns2:
    def __init__(self, config, u_previous_, time_domain_):
        self.config = config
        self.pre_model_save_path = config.pre_model_save_path
        self.save_dir = config.save_path
        self.n_int = config.n_int
        self.n_sb = config.n_sb
        self.n_tb = config.n_tb
        self.nx = config.nx
        self.ny = config.ny
        self.optimizer_name = config.optimizer
        self.device = config.device
        self.u_previous = u_previous_.to(self.device)
        self.time_domain = time_domain_.to(self.device)
        #parameters
        self.E = 210
        self.nu = 0.3
        self.G = (self.E / (2 * (1 + self.nu)))
        self.g = 0.0027
        self.l = 0.1

        # Extrema of the solution domain (x,y) in [0,1]x[0,1]
        self.domain_extrema = torch.tensor([[0, 1],  # x dimension
                                            [0, 1]])  # y dimension
        # Number of space dimensions
        self.space_dimensions = 2

        # F Dense NN to approximate the solution of the underlying heat equation
        self.approximate_solution = NeuralNet(input_dimension=3, output_dimension=3,
                                              n_hidden_layers=4,
                                              neurons=20,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=42).to(self.device)
        '''self.approximate_solution = MultiVariatePoly(self.domain_extrema.shape[0], 3)'''
        if self.pre_model_save_path:
            self.load_checkpoint()

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        self.strueng = StrgridEngine(dimension=2, grid_size=(self.nx, self.ny))

        # Training sets S_sb, S_tb, S_int as torch dataloader
        self.training_set_sb, self.training_set_itb, self.training_set_etb, self.training_set_int = self.assemble_datasets()

        # Optimizer
        self.init_optimizer()

    ################################################################################################
    # Function to linearly transform a tensor whose value are between 0 and 1
    # to a tensor whose values are between the domain extrema
    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    def add_initial_temporal_boundary_points(self):
        # 从时间坐标中获取初始时间
        t_initial = torch.tensor([self.time_domain[0]])
        input_itb_time = torch.tile(t_initial[:, None], [self.n_tb, 1])
        x_0 = torch.linspace(0, 1, 21)
        y_0 = torch.cat([torch.linspace(0, 0.49, 10), torch.linspace(0.51, 1, 10)])
        grid_x0, grid_y0 = torch.meshgrid(x_0, y_0)
        input_itb_0 = torch.stack((grid_x0.reshape(-1), grid_y0.reshape(-1)), dim=1)
        #x_02 = torch.linspace(0.51, 1, 10)
        #y_02 = torch.linspace(0, 1, 21)
        #grid_x02, grid_y02 = torch.meshgrid(x_02, y_02)
        #input_tb_02 = torch.stack((grid_x02.reshape(-1), grid_y02.reshape(-1)), dim=1)
        #input_tb_0 = torch.cat([input_tb_01, input_tb_02], 0)
        # 在空间坐标上均匀取样裂纹部分(初始裂纹部分以及裂纹产生路径部分上取点)
        x_1 = torch.linspace(0, 1, 101)
        y_1 = torch.full((101, 1), 0.5)
        input_itb_1 = torch.cat([x_1.unsqueeze(1), y_1], dim=1)
        input_itb_space = torch.cat([input_itb_0, input_itb_1], 0)
        input_itb = torch.cat((input_itb_space, input_itb_time), dim=1)
        return input_itb

    def add_end_temporal_boundary_points(self):
        # 从时间坐标中获取初始时间
        t_end = torch.tensor([self.time_domain[1]])
        input_etb_time = torch.tile(t_end[:, None], [self.n_tb, 1])
        x_0 = torch.linspace(0, 1, 21)
        y_0 = torch.cat([torch.linspace(0, 0.49, 10), torch.linspace(0.51, 1, 10)])
        grid_x0, grid_y0 = torch.meshgrid(x_0, y_0)
        input_etb_0 = torch.stack((grid_x0.reshape(-1), grid_y0.reshape(-1)), dim=1)
        #x_02 = torch.linspace(0.51, 1, 10)
        #y_02 = torch.linspace(0, 1, 21)
        #grid_x02, grid_y02 = torch.meshgrid(x_02, y_02)
        #input_tb_02 = torch.stack((grid_x02.reshape(-1), grid_y02.reshape(-1)), dim=1)
        #input_tb_0 = torch.cat([input_tb_01, input_tb_02], 0)
        # 在空间坐标上均匀取样裂纹部分(初始裂纹部分以及裂纹产生路径部分上取点)
        x_1 = torch.linspace(0, 1, 101)
        y_1 = torch.full((101, 1), 0.5)
        input_etb_1 = torch.cat([x_1.unsqueeze(1), y_1], dim=1)
        input_etb_space = torch.cat([input_etb_0, input_etb_1], 0)
        input_etb = torch.cat((input_etb_space, input_etb_time), dim=1)
        return input_etb

    # Function returning the input-output tensor required to assemble the training set S_sb corresponding to the spatial boundary
    def add_spatial_boundary_points(self):
        t = torch.linspace(self.time_domain[0], self.time_domain[1], 2)
        input_sb_time = torch.tile(t[:, None], [self.n_sb, 1])
        y0 = self.domain_extrema[1, 0]
        yL = self.domain_extrema[1, 1]

        input_sb_space = self.convert(self.soboleng.draw(self.n_sb))
        input_sb_space_B = torch.clone(input_sb_space)
        input_sb_space_B[:, 1] = y0
        input_sb_space_B = torch.tile(input_sb_space_D, [2, 1])
        input_sb_B = torch.cat((input_sb_space_D, input_sb_time), dim=1) #下边界(bottom boundary)
        input_sb_space_U = torch.clone(input_sb_space)
        input_sb_space_U[:, 1] = yL
        input_sb_space_U = torch.tile(input_sb_space_U, [2, 1])
        input_sb_U = torch.cat((input_sb_space_U, input_sb_time), dim=1) #上边界(upper boundary)
        input_sb = torch.cat([input_sb_U, input_sb_D], 0)

        return input_sb

    def add_interior_points(self):
        t = torch.linspace(self.time_domain[0], self.time_domain[1], 2)
        input_int_time = torch.tile(t[:, None], [self.n_tb, 1])
        input_int_space = self.convert(self.strueng.generate_structure_grid())
        input_int_space = torch.tile(input_int_space, [2, 1])
        input_int = torch.cat((input_int_space, input_int_time), dim=1)
        return input_int

    def assemble_datasets(self):
        input_sb = self.add_spatial_boundary_points()  # S_sb
        input_itb = self.add_initial_temporal_boundary_points()  # S_itb
        input_etb = self.add_end_temporal_boundary_points()  # S_etb
        input_int = self.add_interior_points()  # S_int

        training_set_sb = DataLoader(torch.utils.data.TensorDataset(input_sb),
                                     batch_size=2 * self.space_dimensions * self.n_sb, shuffle=False)

        training_set_itb = DataLoader(torch.utils.data.TensorDataset(input_itb),
                                     batch_size=self.n_tb, shuffle=False)

        training_set_etb = DataLoader(torch.utils.data.TensorDataset(input_itb),
                                     batch_size=self.n_tb, shuffle=False)

        training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int),
                                      batch_size=self.n_int, shuffle=False)

        return training_set_sb, training_set_itb, training_set_etb, training_set_int

    def compute_ic_residual(self, input_tb):
        u1_previous = self.u_previous[:, 0].reshape(-1, 1)
        u2_previous = self.u_previous[:, 1].reshape(-1, 1)
        c_previous = self.u_previous[:, 2].reshape(-1, 1)
        input_tb.requires_grad = True
        u = self.approximate_solution(input_tb)
        u1 = u[:, 0].reshape(-1, 1)
        u2 = u[:, 1].reshape(-1, 1)
        c = u[:, 2].reshape(-1, 1)

        residual_u1 = u1 - u1_previous
        residual_u2 = u2 - u2_previous
        residual_c1 = c - c_previous

        return residual_u1.reshape(-1, ), residual_u2.reshape(-1, ), residual_c1.reshape(-1, )

    def compute_bcU_residual(self, input_sb):
        input_sb.requires_grad = True
        u = self.approximate_solution(input_sb)
        u2 = u[:, 1].reshape(-1, 1)
        residual_U_2 = u2 - 0.001 * input_sb[:, 2]
        return  residual_U_2.reshape(-1, )

    def compute_bcB_residual(self, input_sb):
        input_sb.requires_grad = True
        u = self.approximate_solution(input_sb)
        u1 = u[:, 0].reshape(-1, 1)
        u2 = u[:, 1].reshape(-1, 1)
        residual_B_1 = u1
        residual_B_2 = u2
        return residual_B_1.reshape(-1, ), residual_B_2.reshape(-1, )

    # Function to compute the PDE residuals
    def compute_pde_residual(self, input_int):
        input_int.requires_grad = True
        u = self.approximate_solution(input_int)

        u1 = u[:, 0].reshape(-1, 1)
        u2 = u[:, 1].reshape(-1, 1)
        c = u[:, 2].reshape(-1, 1)
        grad_u1 = torch.autograd.grad(u1.sum(), input_int, create_graph=True)[0]
        u1_1, u1_2, u1_t = grad_u1[:, 0], grad_u1[:, 1], grad_u1[:, 2]
        grad_u1_1 = torch.autograd.grad(u1_1.sum(), input_int, create_graph=True)[0]
        u1_11, u1_12 = grad_u1_1[:, 0], grad_u1_1[:, 1]
        grad_u1_2 = torch.autograd.grad(u1_2.sum(), input_int, create_graph=True)[0]
        u1_21, u1_22 = grad_u1_2[:, 0], grad_u1_2[:, 1]

        grad_u2 = torch.autograd.grad(u2.sum(), input_int, create_graph=True)[0]
        u2_1, u2_2, u2_t = grad_u2[:, 0], grad_u2[:, 1], grad_u2[:, 2]
        grad_u2_1 = torch.autograd.grad(u2_1.sum(), input_int, create_graph=True)[0]
        u2_11, u2_12 = grad_u2_1[:, 0], grad_u2_1[:, 1]
        grad_u2_2 = torch.autograd.grad(u2_2.sum(), input_int, create_graph=True)[0]
        u2_21, u2_22 = grad_u2_2[:, 0], grad_u2_2[:, 1]

        grad_c = torch.autograd.grad(c.sum(), input_int, create_graph=True)[0]
        c_1, c_2, c_t = grad_c[:, 0], grad_c[:, 1], grad_c[:, 2]
        grad_c_1 = torch.autograd.grad(c_1.sum(), input_int, create_graph=True)[0]
        c_11 = grad_c_1[:, 0]
        grad_c_2 = torch.autograd.grad(c_2.sum(), input_int, create_graph=True)[0]
        c_22 = grad_c_2[:, 1]

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

        residual_c = c_t

        return residual_PDE_1.reshape(-1, ), residual_PDE_2.reshape(-1, ), residual_PDE_3.reshape(-1, ), residual_c.reshape(-1, )

    # Function to compute the total loss (weighted sum of spatial boundary loss, temporal boundary loss and interior loss)
    def compute_loss(self, inp_train_sb,  inp_train_itb,  inp_train_int, verbose=True):

        r_int_1, r_int_2, r_int_3, r_int_c = self.compute_pde_residual(inp_train_int)
        r_sbU_u2 = self.compute_bcU_residual(inp_train_sb[0:2*self.n_sb, :])
        r_sbB_u1, r_sbB_u2 = self.compute_bcB_residual(inp_train_sb[2*self.n_sb:, :])
        r_tb_u1, r_tb_u2, r_tb_c, = self.compute_ic_residual(inp_train_itb)

        loss_sb = torch.mean(abs(r_sbU_u2) ** 2) + torch.mean(abs(r_sbB_u2) ** 2) + torch.mean(abs(r_sbB_u1) ** 2)
        loss_tb = torch.mean(abs(r_tb_u1) ** 2) + torch.mean(abs(r_tb_u2) ** 2) + torch.mean(abs(r_tb_c) ** 2)

        loss_int = torch.mean(abs(r_int_1) ** 2) + torch.mean(abs(r_int_2) ** 2) + torch.mean(abs(r_int_3) ** 2)
        loss_inequality = torch.mean(torch.relu(-r_int_c))
        loss = loss_sb + loss_tb + loss_int + loss_inequality

        if verbose: print("Total loss: ", round(torch.log10(loss).item(), 4),
                          "| Inequality Loss: ", round(torch.log10(loss_inequality).item(), 4),
                          "| BC Loss: ", round(torch.log10(loss_sb).item(), 4),
                          "| IC Loss: ", round(torch.log10(loss_tb).item(), 4),
                          "| PDE Loss: ", round(torch.log10(loss_int).item(), 4))

        return loss, loss_sb, loss_tb, loss_int, loss_inequality

    def init_optimizer(self):
        '''Initialize optimizer'''
        if self.optimizer_name == "lbfgs":
            self.optimizer = torch.optim.LBFGS(self.approximate_solution.parameters(), lr=float(0.5),
                                               max_iter=self.config.max_iter,
                                               max_eval=50000, tolerance_change=1.0 * np.finfo(float).eps,
                                               history_size=150, line_search_fn="strong_wolfe")
        elif self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.approximate_solution.parameters(), lr=self.config.lr)
        else:
            raise NotImplementedError(f"Optimizer {self.optimizer_name} not implemented")

        # init scheduler
        if self.config.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                             step_size=self.config.scheduler_step_size,
                                                             gamma=self.config.scheduler_gamma)
            if self.config.optimizer == "lbfgs":
                raise NotImplementedError(f"Scheduler not implemented for optimizer {self.config.optimizer}")
        else:
            self.scheduler = None

    def fit(self, num_epochs, max_iter, lr, verbose=True):
        '''Train process'''
        # Training preparation
        self.approximate_solution.train()
        best_loss, best_epoch, best_state = np.inf, -1, None
        losses = []
        losses_sb = []
        losses_tb = []
        losses_int = []
        losses_inequality = []

        epoch = {
            "lbfgs": max_iter,
            "adam": num_epochs
        }[self.optimizer_name]

        # pbar = tqdm(range(epoch), desc = 'Epoch', colour='blue')
        def train_batch(batch_sb, batch_itb, batch_int):
            inp_train_sb = batch_sb[0].to(self.device)
            inp_train_itb = batch_itb[0].to(self.device)
            inp_train_int = batch_int[0].to(self.device)

            def closure():
                self.optimizer.zero_grad()
                loss, loss_sb, loss_tb, loss_int, loss_inequality = self.compute_loss(
                    inp_train_sb, inp_train_itb, inp_train_int, verbose=verbose)
                # backpropragation
                loss.backward()
                # recording
                losses.append(loss.item())
                losses_sb.append(loss_sb.item())
                losses_tb.append(loss_sb.item())
                losses_int.append(loss_int.item())
                losses_inequality.append(loss_inequality.item())

                if self.config.optimizer == "lbfgs":
                    pbar.set_postfix(loss=losses[-1])
                    pbar.update(1)
                return loss
            return closure

        # training
        if self.optimizer_name == "lbfgs":
            pbar = tqdm(total=self.config.max_iter, desc='Batch',
                        colour='blue')  # Progress bar for LBFGS based on batches
            # optimizer = torch.optim.LBFGS(self.approximate_solution.parameters(), lr=lr, max_iter=max_iter, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)

            for j, (batch_sb, batch_itb, batch_etb, batch_int) in enumerate(zip(
                    self.training_set_sb, self.training_set_itb, self.training_set_etb, self.training_set_int)):
                self.optimizer.step(closure=train_batch(batch_sb, batch_itb, batch_int))
            pbar.close()
            self.save_checkpoint()

        elif self.optimizer_name == "adam":
            pbar = tqdm(total=num_epochs, desc='Epoch', colour='blue')  # Progress bar for Adam based on epochs
            # optimizer = torch.optim.Adam(self.approximate_solution.parameters(), lr=lr)
            for ep in range(num_epochs):
                for j, (batch_sb, batch_itb, batch_etb, batch_int) in enumerate(zip(
                        self.training_set_sb, self.training_set_itb, self.training_set_etb, self.training_set_int)):

                    train_batch(batch_sb, batch_itb, batch_int)()
                    self.optimizer.step()
                    if self.config.use_scheduler:
                        self.scheduler.step()

                    # save model
                    if losses[-1] < best_loss:
                        best_epoch = ep
                        best_loss = losses[-1]
                        best_state = deepcopy(self.approximate_solution.state_dict())
                        best_loss = losses[-1]
                pbar.set_postfix(loss=losses[-1])
                pbar.update(1)
            pbar.close()

            self.approximate_solution.load_state_dict(best_state)
            self.save_checkpoint()

        with torch.no_grad():
            u_end = self.approximate_solution(list(self.training_set_etb)[0][0].to(self.device))
            #solution = torch.cat((list(self.training_set_etb)[0][0], u_end), 1)
            #np.savetxt('./result_0.1.txt', solution.cpu().numpy(), delimiter=',', header='t,x,y,u1,u2,c')
            #output_filename = f"output_0.1.csv"
            #with open(output_filename, mode='w', newline='') as file:
                #writer = csv.writer(file)
                #writer.writerows(solution.cpu().numpy())

        # plot losses vs epoch
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(np.arange(len(losses)), losses, label="loss")
        ax.plot(np.arange(len(losses_sb)), losses_sb, label="loss_sb")
        ax.plot(np.arange(len(losses_tb)), losses_sb, label="loss_tb")
        ax.plot(np.arange(len(losses_int)), losses_int, label="loss_int")
        ax.plot(np.arange(len(losses_inequality)), losses_inequality, label="loss_inequality")
        if best_epoch != -1:
            ax.scatter([best_epoch], [best_loss], c='r', marker='o', label="best loss")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss vs Epoch')
        ax.legend()
        ax.set_xlim(left=0)
        ax.set_yscale('log')
        plt.savefig(f'loss.png')

        return u_end

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
