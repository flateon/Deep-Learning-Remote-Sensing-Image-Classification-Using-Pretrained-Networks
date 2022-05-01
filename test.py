import torch
from tqdm import tqdm


def test(model_list, test_dataloader):
    with torch.no_grad():
        correct, total = 0, 0
        pretrained_model, model = model_list[0].eval(), model_list[1].eval()
        for i, (images, targets) in enumerate(tqdm(test_dataloader, leave=False)):
            targets, images = targets.to('cuda'), images.to('cuda')
            out = model(pretrained_model(images))
            predict = out.argmax(1)
            correct += (predict == targets).sum()
            total += targets.shape[0]
    return 100. * correct.item() / total
