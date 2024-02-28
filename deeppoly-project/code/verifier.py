import argparse
import torch

from networks import get_network
from utils.loading import parse_spec
from deeppoly import DeepPolyVerifier

DEVICE = "cpu"


def analyze_backprop(net, inputs, eps, true_label):
    change_threshold = 0.001
    EPOCHS = 20000
    STEP_REINIT = 70
    verifier = DeepPolyVerifier(net, true_label, inputs)

    optimizer = torch.optim.SGD([a for a in verifier.alphas], lr=100)

    for params in verifier.parameters():
        params.requires_grad = True

    for alpha in verifier.alphas:
        alpha.requires_grad = True

    verifier.train()
    for i in range(EPOCHS):
        optimizer.zero_grad()
        classes = verifier(inputs, eps)

        # print(torch.min(classes))
        if torch.all(classes > 0):
            return True

        loss = torch.relu(-classes).sum()
        loss.backward(loss)

        optimizer.step()

        if (i + 1) % STEP_REINIT == 0:
            for alpha in verifier.alphas:
                torch.nn.init.uniform_(alpha.data, a=0.0, b=1.0)
            for g in optimizer.param_groups:
                g['lr'] = 0.9

    return torch.all(classes > 0)

def analyze_single(net, inputs, eps, true_label):
    verifier = DeepPolyVerifier(net, true_label, inputs)
    classes = verifier(inputs, eps)
    return torch.all(classes > 0)

def analyze(net, inputs, eps, true_label):
    conv = False
    backprop = False
    for layer in net:
        if isinstance(layer, torch.nn.ReLU):
            backprop = True
        elif isinstance(layer, torch.nn.LeakyReLU):
            backprop = True
        if isinstance(layer, torch.nn.Conv2d):
            conv = True

    if conv and backprop:
        return False

    if not(backprop):
        return analyze_single(net, inputs, eps, true_label)
    else:
        return analyze_backprop(net, inputs, eps, true_label)


def main():
    parser = argparse.ArgumentParser(
        description="Neural network verification using DeepPoly relaxation."
    )
    parser.add_argument(
        "--net",
        type=str,
        choices=[
            "fc_base",
            "fc_1",
            "fc_2",
            "fc_3",
            "fc_4",
            "fc_5",
            "fc_6",
            "fc_7",
            "conv_base",
            "conv_1",
            "conv_2",
            "conv_3",
            "conv_4",
        ],
        required=True,
        help="Neural network architecture which is supposed to be verified.",
    )
    parser.add_argument("--spec", type=str, required=True, help="Test case to verify.")
    args = parser.parse_args()

    true_label, dataset, image, eps = parse_spec(args.spec)

    # print(args.spec)

    net = get_network(args.net, dataset, f"models/{dataset}_{args.net}.pt").to(DEVICE)

    image = image.to(DEVICE)
    out = net(image.unsqueeze(0))

    pred_label = out.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, image, eps, true_label):
        print("verified")
    else:
        print("not verified")


if __name__ == "__main__":
    main()
