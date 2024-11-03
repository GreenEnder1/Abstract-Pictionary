import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from typing import List # for type hinting

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str] = None,
                        transform=None,
                        device=device):
    '''
    makes a prediciton on a target image on a trained model, and plots the image and prediciton
    '''
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)
    target_image = target_image/255 # this changes read_image output to be between 0 and 1

    if transform:
      target_image = transform(target_image)

    model.to(device)

    model.eval()
    with torch.inference_mode():
      target_image = target_image.unsqueeze(0) # add one for batch

      target_image_pred = model(target_image.to(device))
      target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
      target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)


      plt.imshow(target_image.squeeze().permute(1,2,0)) # squeeze to remove batchsize, and permuted for matplotlib
      if class_names:
        title = f'Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}'
      else:
        title = f'Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}'
      plt.title(title)
      plt.axis(False)


model = torch.load('trained_models/model_0.pth', weights_only=False)
model_state_dict = model['model_state_dict']
class_names = model['class_names']

model.eval()

custom_image_transform = transforms.Compose([transforms.Resize(size=(600,600))])

custom_image_path = '../dark-grey-background-phpf8ys7h4fgijgt-3755329406.jpg'
pred_and_plot_image(model=model,
                    image_path=custom_image_path,
                    class_names=class_names,
                    transform=custom_image_transform,
                    device=device)