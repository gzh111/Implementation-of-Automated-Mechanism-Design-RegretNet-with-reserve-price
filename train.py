from argparse import ArgumentParser
import torch
import numpy as np
import matplotlib.pyplot as plt
from regretnet.datasets import generate_dataset_1x2, generate_dataset_nxk
from regretnet.regretnet import RegretNet, train_loop, test_loop
from torch.utils.tensorboard import SummaryWriter
from regretnet.datasets import Dataloader
from util import plot_12_model, plot_payment, plot_loss, plot_regret
import json
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = ArgumentParser()
parser.add_argument('--random-seed', type=int, default=0)
parser.add_argument('--num-examples', type=int, default=100000)
parser.add_argument('--test-num-examples', type=int, default=3000)
parser.add_argument('--n-agents', type=int, default=1)
parser.add_argument('--n-items', type=int, default=2)
parser.add_argument('--reserved-price', type=float, default=0)
parser.add_argument('--num-epochs', type=int, default=500)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--test-batch-size', type=int, default=100)
parser.add_argument('--model-lr', type=float, default=1e-3)
parser.add_argument('--misreport-lr', type=float, default=0.1)
parser.add_argument('--misreport-iter', type=int, default=25)
parser.add_argument('--test-misreport-iter', type=int, default=1000)
parser.add_argument('--rho', type=float, default=50.0)
parser.add_argument('--rho-incr-iter', type=int, default=6.0)
parser.add_argument('--rho-incr-amount', type=float, default=5.0)
# parser.add_argument('--rho-ir', type=float, default=1.0)
# parser.add_argument('--rho-incr-iter-ir', type=int, default=5)
# parser.add_argument('--rho-incr-amount-ir', type=float, default=5.0)
parser.add_argument('--payment_power', type=float, default=0.)
parser.add_argument('--lagr_update_iter', type=int, default=6.0)
parser.add_argument('--lagr_update_iter_ir', type=int, default=6.0)
parser.add_argument('--lagr_update_iter_rp', type=int, default=6.0)
parser.add_argument('--ir_penalty_power', type=float, default=2)
parser.add_argument('--resume', default="")

# architectural arguments
parser.add_argument('--p_activation', default='full_relu_clipped')
parser.add_argument('--a_activation', default='softmax')
parser.add_argument('--hidden_layer_size', type=int, default=100)
parser.add_argument('--n_hidden_layers', type=int, default=2)
parser.add_argument('--separate', action='store_true')
parser.add_argument('--rs_loss', action='store_true')

parser.add_argument('--teacher_model', default="")
parser.add_argument('--name', default='testing_name')

if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    writer = SummaryWriter(log_dir=f"run/{args.name}",
                           comment=f"{args}")


    model = RegretNet(args.n_agents, args.n_items, activation='relu', hidden_layer_size=args.hidden_layer_size,
                      n_hidden_layers=args.n_hidden_layers, p_activation=args.p_activation,
                      a_activation=args.a_activation, separate=args.separate).to(DEVICE)
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    if args.teacher_model != "":
        checkpoint = torch.load(args.teacher_model)
        teachermodel = RegretNet(**checkpoint['arch'])
        teachermodel.load_state_dict(checkpoint['state_dict'], strict=False)
        teachermodel.to(DEVICE)
    else:
        teachermodel=None

    train_data = generate_dataset_nxk(args.n_agents, args.n_items, args.num_examples).to(DEVICE)
    train_loader = Dataloader(train_data, batch_size=args.batch_size, shuffle=True)
    test_data = generate_dataset_nxk(args.n_agents, args.n_items, args.test_num_examples).to(DEVICE)
    test_loader = Dataloader(test_data, batch_size=args.test_batch_size, shuffle=True)

    train_loop(
        model, train_loader, test_loader, args, device=DEVICE, writer=writer
    )
    writer.close()
    pay, alloc, pay_result, result = test_loop(model, test_loader, args, device=DEVICE)
    print(f"Experiment:{args.name}")
    print(json.dumps(result, indent=4, sort_keys=True))
    # plot_payment(model, grid_width=0.01, name=args.name)
    # plot_12_model(model, grid_width=0.01, name=args.name)
