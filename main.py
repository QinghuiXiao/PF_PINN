import argparse
from Trainer.methods import PINN, DPINN

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trainer', type=str, default="PINN")
    parser.add_argument('--device', type=str, default="cuda", help="device: cpu or cuda:id")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_path', type=str, default='./lbfgs_10000.pt')
    parser.add_argument('--pre_model_save_path', type=str, default=None)

    # Training setup
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--iters', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--max_iter', type=int, default=5000, help="maximum number of iterations for lbfgs")
    parser.add_argument('--use_scheduler', action='store_true', help="use learning rate scheduler")
    parser.add_argument('--scheduler_step_size', type=int, default=500, help="step size for learning rate scheduler")
    parser.add_argument('--scheduler_gamma', type=float, default=0.9, help="gamma for learning rate scheduler")

    # PINN setup
    parser.add_argument('--n_int', type=int, default=10201)
    parser.add_argument('--n_sb', type=int, default=200)
    parser.add_argument('--n_tb', type=int, default=2651)
    parser.add_argument('--nt', type=int, default=10)
    parser.add_argument('--nx', type=int, default=51)
    parser.add_argument('--ny', type=int, default=51)
    parser.add_argument('--delta_t', type=float, default=0.1)

    # Network setup
    parser.add_argument('--neurons', type=int, default=20)
    parser.add_argument('--n_hidden_layers', type=int, default=4)

    args = parser.parse_args()
    assert args.train or args.test, "Please specify --train or --test"

    Trainer = {"PINN": PINN,
               "DPINN": DPINN
               }[args.trainer]

    if args.train:
        Trainer(args)