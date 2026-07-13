import torch
from torch.autograd import Variable
import numpy as np
from .meta_model import ResModel

def meta_testing(train_loader, test_loader, model, backbone, num_class):
    meta_lr=0.01
    criterion = torch.nn.CrossEntropyLoss().cuda()
    meta_loss=0
    meta_psl_loss=0
    test_net = ResModel(backbone, output_dim=num_class).cuda()
    test_net.copy(model, same_var=True)
    meta_optimizer = torch.optim.SGD(test_net.get_params(meta_lr))

    test_net.train()
    for i in range(1):
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.float().cuda(), y.long().cuda()

                #Train on Du
                u_prob = torch.softmax(test_net(x), dim=1)
                u_pred = u_prob.max(1)
                
                u_mask = u_pred[0] >= 0
                im_u_1 = x[u_mask]
                psl_u = u_pred[1][u_mask]

                out_mix = test_net(im_u_1)
                meta_loss= criterion(out_mix, psl_u)
                meta_loss = Variable(meta_loss, requires_grad = True)
                alpha = 1
                lam = np.random.beta(alpha, alpha)
                
                test_net.zero_grad()
                meta_optimizer.zero_grad()
                meta_loss.backward()
                meta_optimizer.step()

    ### Inference ###
    test_net.eval()
    acc, cnt = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.float().cuda(), y.long().cuda()
            out = test_net(x)
            pred = out.argmax(dim=1)
            acc += (pred == y).float().sum().item()
            cnt += len(x)
    return 100 * acc / cnt

def evaluation(loader, model):
    model.eval()
    acc, cnt = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.float().cuda(), y.long().cuda()
            out = model(x)
            pred = out.argmax(dim=1)
            acc += (pred == y).float().sum().item()
            cnt += len(x)
    model.train()
    return 100 * acc / cnt


def prediction(loader, model):
    model.eval()
    P, F = [], []
    with torch.no_grad():
        for x, _ in loader:
            x = x.cuda().float()
            F.append(model.get_features(x))
            P.append(model.get_predictions(F[-1]))
    model.train()
    return torch.vstack(P), torch.vstack(F)
