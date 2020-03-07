import time, argparse, torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import Net
from util import load_data
import torch.backends.cudnn as cudnn


def parser_args():
  parser = argparse.ArgumentParser(description='Ablation study')
  parser.add_argument('--gpu', type=int, default=0,
                      help='gpu instance to use (default: 0)')
  parser.add_argument('--seed', type=int, default=1818,
                      help='random seed (default: 1818)')

  # dataset
  parser.add_argument('--dataset', type=str, default='cifar10',
                      choices={'cifar10', 'cifar100', 'svhn'}, 
                      help='dataset (default: cifar10)')
  parser.add_argument('--data_path', type=str, 
                      help='path to load dataset')
  parser.add_argument('--nclass', type=int, default=10,
                      help='number of classes (default: 10)')
  parser.add_argument('--batchsize', type=int, default=128,
                      help='batch size (default: 128)')
  parser.add_argument('--normalize', action='store_true', 
                      help='whether to normalize data')

  # optimization
  parser.add_argument('--optim', type=str, default='sgd',
                      help='optimizer (default: sgd)')
  parser.add_argument('--lr', type=float, default=1e-3,
                      help='learning rate (default: 1e-3)')
  parser.add_argument('--wd', type=float, default=1e-6,
                      help='weight decay (default: 5e-5)')
  parser.add_argument('--niter', type=int, default=80000,
                      help='number of training iterations (default: 80000)')
  parser.add_argument('--stepsize', type=int, default=20000,
                      help='by which learning rate is halved (default: 20000)')

  # network
  parser.add_argument('--fnet_path', type=str, 
                      help='path to load fnet')
  parser.add_argument('--hnet_path', type=str, 
                      help='path to load hnet')
  parser.add_argument('--clf_path', type=str, 
                      help='path to load clf')
  parser.add_argument('--model_path', type=str, 
                      help='path to save model')
  parser.add_argument('--freeze_hnet', nargs='+', type=int, default=0,
                      help='hnet layers to freeze')
  parser.add_argument('--linearize_hnet', nargs='+', type=int, default=0, 
                      help='hnet layers to linearize')
  parser.add_argument('--linearize_clf', action='store_true',
                      help='whether to linearize the classifier')

  return parser.parse_args()


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def train(device, loader, model, optimizer, niter, stepsize, losses, it=0):
  batch_time = AverageMeter()
  data_time = AverageMeter()
  end = time.time()

  curr_iter = it
  model.train()

  for (x, y) in loader:
    data_time.update(time.time() - end)

    # update learning rate
    if curr_iter != 0 and curr_iter % stepsize == 0:
      for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.5
        print('iter %d learning rate is %.5f' % (curr_iter, param_group['lr']))

    x, y = x.to(device), y.to(device)
    
    ### proposed model
    logits, jvp = model(x)
    logits = logits + jvp

    ### gradient baseline (second term in proposed model)
    # _, jvp = model(x)
    # logits = jvp

    ### activation baseline (first term in proposed model)
    ### fine-tuning
    # logits = model(x)

    loss = nn.CrossEntropyLoss()(logits, y)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 10)  # clip gradient
    optimizer.step()

    losses.update(loss.item(), x.size(0))
    batch_time.update(time.time() - end)
    end = time.time()

    if curr_iter % 50 == 0:
      print('Iteration[{0}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
             curr_iter, 
             batch_time=batch_time, data_time=data_time, loss=losses))
    curr_iter += 1

    if curr_iter == niter:
      break

  return curr_iter


def evaluate(device, loader, model):
  model.eval()
  ncorr = 0

  for i, (x, y) in enumerate(loader):
    x, y = x.to(device), y.to(device)

    with torch.no_grad():
      ### proposed model
      logits, jvp = model(x)
      logits = logits + jvp

      ### gradient baseline (second term in proposed model)
      # _, jvp = model(x)
      # logits = jvp

      ### activation baseline (first term in proposed model)
      ### fine-tuning
      # logits = model(x)

      pred = torch.argmax(logits.detach_(), dim=1)
      ncorr += (pred == y).sum()

  acc = ncorr.float() / len(loader)
  print(acc.item())


def main():
  args = parser_args()
  print('Batch size: %d' % args.batchsize)
  print('Initial learning rate: %.5f' % args.lr)
  print('Weight decay: %.6f' % args.wd)

  device = torch.device('cuda:' + str(args.gpu) 
    if torch.cuda.is_available() else 'cpu')
  cudnn.benchmark = True

  # fix random seed
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)
  np.random.seed(args.seed)

  net = Net(nclasses=args.nclass)
  fnet = torch.load(args.fnet_path)  # feature net (theta_1)
  hnet = torch.load(args.hnet_path)  # head net (theta_2)
  clf = torch.load(args.clf_path)    # classifier (omega)
  
  # load parameters (random vs. pre-trained) as appropriate
  net.load_fnet(fnet, freeze=True)
  net.load_hnet(hnet, reinit_idx=(), 
    freeze_idx=args.freeze_hnet, linearize_idx=args.linearize_hnet)
  net.load_clf(clf, reinit=False, linearize=args.linearize_clf)
  net.to(device)

  params = list(filter(lambda p: p.requires_grad, net.parameters()))
  if args.optim == 'sgd':
    optimizer = optim.SGD(
      params, lr=args.lr, weight_decay=args.wd, momentum=0.9)
  elif args.optim == 'adam':
    optimizer = optim.Adam(
      params, lr=args.lr, betas=(0.5, 0.999), weight_decay=args.wd)

  # check trainable parameters
  for p in params:
    print(p.size()) 

  # load training and test data
  train_loader, test_loader = load_data(
    args.dataset, args.data_path, args.batchsize, args.normalize)
  
  print('----- Training phase -----')
  it = 0
  losses = AverageMeter()

  while it < args.niter:
    it = train(
      device, train_loader, net, optimizer, 
      args.niter, args.stepsize, losses, it=it)

  print('----- Evaluation phase -----')
  print('> test accuracy:')
  evaluate(device, test_loader, net)
  torch.save(net.cpu(), args.model_path)


if __name__ == '__main__':
  main()