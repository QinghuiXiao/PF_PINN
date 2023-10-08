import torch
import numpy as np  
from .base import Pinns, Pinns2
import csv

def PINN(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    PINN = Pinns(config=args)
    hist = PINN.fit(num_epochs=args.epochs, max_iter=args.iters, lr=args.lr, verbose=False)
    #PINN.plotting()

def DPINN(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    networks = []
    time_intervals = []
    u_previous = torch.zeros(args.n_tb, 3)
    u_previous[420:471, 2] = 1 #裂纹上取的点c=1

    for i in range(0, args.nt):
        time_domain = torch.tensor([i * args.delta_t, (i+1) * args.delta_t])  # t dimension
        time_intervals.append((i*args.delta_t, (i+1)*args.delta_t))
        DPINN = Pinns2(config=args, u_previous_=u_previous, time_domain_=time_domain)

        print(f"Training network for time interval [{i * args.delta_t}, {(i + 1) * args.delta_t}]")
        u_end = DPINN.fit(num_epochs=args.epochs, max_iter=args.iters, lr=args.lr,
                                  verbose=False)  # Fit the PINN
        u_previous = u_end  # Update the initial condition for the next time interval
        networks.append(DPINN)
        print(f"Training network for time interval [{i * args.delta_t}, {(i + 1) * args.delta_t}] complete")

        # Save u_previous to i_output.csv
        output_filename = f"{i+1}_output.csv"
        with open(output_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(u_previous.cpu().numpy())