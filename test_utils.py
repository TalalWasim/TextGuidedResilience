import torch
import torch.nn.functional as F

def test(net, test_loader):
  """Evaluate network on given dataset."""
  net.eval()
  total_loss = 0.
  total_correct = 0
  with torch.no_grad():
    for images, targets in test_loader:
      images, targets = images.cuda(), targets.cuda()
      logits = net(images)
      loss = F.cross_entropy(logits, targets)
      pred = logits.data.max(1)[1]
      total_loss += float(loss.data)
      total_correct += pred.eq(targets.data).sum().item()

  return total_loss / len(test_loader.dataset), total_correct / len(test_loader.dataset)

def evaluate(net, val_loader):
    test_loss, test_acc1 = test(net, val_loader)
    print('Test Loss {:.3f} | Test Acc1 {:.3f}'.format(test_loss, 100 * test_acc1))