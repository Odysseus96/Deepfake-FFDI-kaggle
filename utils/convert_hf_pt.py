import timm
import torch

def convert(model_name, output_path):
    model = timm.create_model(model_name, pretrained=True, num_classes=2)
    torch.save(model.state_dict(), output_path)
    print(f'Model weights saved to {output_path}')


if __name__ == '__main__':
    model_name = "efficientnet_b1"
    output_path = "efficientnet_b1.pth"
    convert(model_name, output_path)