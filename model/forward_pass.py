import torch
from torch import nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from typing import List
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TinyVGG(nn.Module):
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
      super().__init__()
      self.conv_block_1 = nn.Sequential(
          nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces spatial dimensions
      )
      self.conv_block_2 = nn.Sequential(
          nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2, stride=2)  # Further reduces spatial dimensions
      )
      
      # Use a dummy input to calculate the output size
      dummy_input = torch.randn(1, input_shape, 64, 64)  # Adjust 64x64 to your input image size
      with torch.no_grad():
          output = self.conv_block_1(dummy_input)
          output = self.conv_block_2(output)
          flattened_output_size = output.view(-1).shape[0]

      self.classifier = nn.Sequential(
          nn.Flatten(),
          nn.Linear(in_features=flattened_output_size, out_features=output_shape)
      )

  def forward(self, x):
      x = self.conv_block_1(x)
    #   print(x.shape)
      x = self.conv_block_2(x)
    #   print(x.shape)
      x = self.classifier(x)
    #   print(x.shape)
      return x


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
    return target_image_pred_probs


def scoring_main(image):

  model_info = torch.load('model/trained_models/model_0.pth', weights_only=False)
  model_state_dict = model_info['model_state_dict']
  class_names = model_info['class_names']

  model_inst = TinyVGG(input_shape=3,
                    hidden_units=10,
                    output_shape=len(class_names)).to(device)

  model_inst.load_state_dict(model_state_dict)
  model_inst.eval()

  custom_image_transform = transforms.Compose([transforms.Resize(size=(64,64))])

  custom_image_path = image
  target_image_pred_probs = pred_and_plot_image(model=model_inst, 
                                                image_path=custom_image_path,
                                                class_names=class_names,
                                                transform=custom_image_transform,
                                                device=device)

  data_dict = dict(zip(class_names, target_image_pred_probs.tolist()[0]))
  data_dict = dict(sorted(data_dict.items(), key=lambda item: item[1], reverse = True))
  scores = json.dumps(data_dict, indent=4)

  print(scores)

  return scores

if __name__ == '__main__':
   scoring_main('data/test/dread/39.jpg')