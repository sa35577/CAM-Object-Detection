import torch
import torchvision.models as models
import cv2
import numpy as np
from torchvision import transforms
import torch.nn as nn


import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

# Define transformations for data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a uniform size
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
])

# Load train and test datasets
train_dataset = ImageFolder('Dataset/train', transform=transform)
test_dataset = ImageFolder('Dataset/test', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



import torchvision.models as models
import torch.nn as nn

# Load a pre-trained ResNet18 model
model = models.resnet18(pretrained=False)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # 2 output classes: face and no-face

criterion = nn.CrossEntropyLoss()


# Training loop
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps')
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
num_epochs = 20
train_correct = 0
train_total = 0
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_total += labels.size(0)
        train_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()

        train_loss += loss.item() * images.size(0)

    epoch_loss = train_loss / len(train_dataset)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_correct / train_total * 100:.2f}%')

# Evaluation on test set
model.eval()
test_correct = 0
test_total = 0
test_loss_total = 0.0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        test_loss_total += loss.item() * images.size(0)


accuracy = test_correct / test_total
test_loss = test_loss_total / len(test_dataset)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%')

# Save the model
torch.save(model, 'model.pth')




# Load and preprocess the image
original_img = cv2.imread('sat.png')
# original_img = cv2.imread('2 faces.png')
# img = cv2.imread('river_hand.jpeg')
# img = cv2.imread('image_2.jpg')
# img = cv2.imread('tejas.jpg')
# img = cv2.imread('shahan.jpg')
# img = cv2.imread('osama.jpg')
# img = cv2.imread('Human1250 copy.png')

if original_img is not None:
    print("Image loaded successfully!")
else:
    print("Unable to load the image. Please check the file path.")

features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

model._modules.get('layer4').register_forward_hook(hook_feature)

img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_img = preprocess(img).unsqueeze(0).to(device)


# Forward pass to get feature maps
with torch.no_grad():
    feature_maps = model(input_img)

params = list(model.parameters())
weight = np.squeeze(params[-2].data.cpu().numpy())
cam = weight[0].dot(features_blobs[0].reshape(-1, 7 * 7))

print("cam", cam)
cam = cam.reshape(7, 7)
cam = cam - np.min(cam)
cam = cam / np.max(cam)
cam = np.uint8(255 * cam)
# cam = cv2.resize(cam, (256, 256))
cam = cv2.resize(cam, (img.shape[1], img.shape[0])) 
print("shape", cam.shape)

# Apply heatmap on the original image
heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
result = heatmap * 0.3 + original_img * 0.5
cv2.imwrite('CAM3.jpg', result)
