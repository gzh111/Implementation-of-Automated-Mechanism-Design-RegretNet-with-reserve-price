import torch
import matplotlib.pyplot as plt

def myerson_auction_uniform_revenue(n):
    '''Gives the expected revenue from a Myerson auction for n agents, 1 item,
    where all agents have valuations distributed Unif[0,1]. The Myerson auction
    is a second-price auction with reserve price of 1/2.'''
    return -( (1.0 - 2.0**(-n) - n) / (1 + n))


def plot_12_model(model, grid_width=0.1, item1_range=(0, 1), item2_range=(0, 1), name="test"):
    item1_min, item1_max = item1_range
    item2_min, item2_max = item2_range
    xcoords = (item1_max - item1_min) * torch.arange(0, 1, grid_width).view(1, -1) + item1_min
    ycoords = (item2_max - item2_min) * torch.arange(0, 1, grid_width).view(-1, 1) + item2_min
    xlen = ycoords.shape[0]
    ylen = xcoords.shape[1]

    xcoords_tiled = xcoords.repeat(xlen, 1)
    ycoords_tiled = ycoords.repeat(1, ylen)

    combined = torch.stack((xcoords_tiled, ycoords_tiled), dim=2)

    output_allocs, output_payments = model(combined.view(-1, 1, 2).cuda())

    output_item1_allocs = output_allocs[:, :, 0].view(xlen, ylen)
    output_item2_allocs = output_allocs[:, :, 1].view(xlen, ylen)
    f, (ax1, ax2) = plt.subplots(2, 1)
    im1 = ax1.imshow(output_item1_allocs.cpu().detach().numpy(), origin='lower', cmap='YlOrRd')
    ax1.set_title('prob of allocating item 1')
    ax1.set_xlabel('v1')
    ax1.set_ylabel('v2')
    im2 = ax2.imshow(output_item2_allocs.cpu().detach().numpy(), origin='lower', cmap='YlOrRd')
    ax2.set_title('prob of allocating item 2')
    ax2.set_xlabel('v1')
    ax2.set_ylabel('v2')
    f.tight_layout()

    plt.savefig(f'result/{name}_model12.png')
    plt.close()

def plot_payment(model, grid_width=0.1, item1_range=(0, 1), item2_range=(0, 1), name="test"):
    item1_min, item1_max = item1_range
    item2_min, item2_max = item2_range
    xcoords = (item1_max - item1_min) * torch.arange(0, 1, grid_width).view(1, -1) + item1_min
    ycoords = (item2_max - item2_min) * torch.arange(0, 1, grid_width).view(-1, 1) + item2_min
    xlen = ycoords.shape[0]
    ylen = xcoords.shape[1]

    xcoords_tiled = xcoords.repeat(xlen, 1)
    ycoords_tiled = ycoords.repeat(1, ylen)

    combined = torch.stack((xcoords_tiled, ycoords_tiled), dim=2)

    _, output_payments = model(combined.view(-1, 1, 2).cuda())

    ax1 = plt.subplot()
    im1 = ax1.imshow((output_payments.view(xlen, ylen)).cpu().detach().numpy(), origin='lower', cmap='YlOrRd')
    ax1.set_title('payment charged')
    ax1.set_xlabel('v1')
    ax1.set_ylabel('v2')

    plt.savefig(f'result/{name}_payment.png')
    plt.close()

def plot_regret(regret, name="test"):
    plt.plot(regret)
    plt.savefig(f'result/{name}_regret.png')
    plt.close()

def plot_loss(loss, name="test"):
    plt.plot(loss)

    plt.savefig(f'result/{name}_loss.png')
    plt.close()