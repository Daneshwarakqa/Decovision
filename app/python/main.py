import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch 
from torchvision import models, transforms
import torch.hub
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

# Load the uploaded image
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Display the uploaded image
image_path = 'room.png'  # Change this to the path of your uploaded image
image = load_image(image_path)
plt.imshow(image)
plt.axis('off')
plt.show()

# Load a pre-trained DeepLabV3 model for segmentation
model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

# Image transformation to fit the model's requirements
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Perform segmentation
def segment_image(image):
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    return output_predictions.numpy()

# Segment the image and display the segmentation mask
segmented = segment_image(image)
plt.imshow(segmented)
plt.axis('off')
plt.show()

# Load MiDaS model
midas = torch.hub.load("isl-org/MiDaS", "MiDaS_small")
midas_transforms = torch.hub.load("isl-org/MiDaS", "transforms").small_transform

# Depth estimation function
def estimate_depth(image):
    input_batch = midas_transforms(image).unsqueeze(0)

    with torch.no_grad():
        depth = midas(input_batch)
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=image.shape[:2],
        mode="bicubic",
        align_corners=False
    ).squeeze().cpu().numpy()
    return depth

# Get depth map and display it
depth_map = estimate_depth(image)
plt.imshow(depth_map, cmap="plasma")
plt.axis('off')
plt.show()

# Basic 3D Visualization with Depth
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.arange(depth_map.shape[1]), np.arange(depth_map.shape[0]))
ax.plot_surface(X, Y, -depth_map, rstride=1, cstride=1, cmap='plasma', edgecolor='none')
plt.show()

# Load Image
image = load_image('Figure_1.png')

# Segment the Image
segmented = segment_image(image)

# Estimate Depth
depth_map = estimate_depth(image)

# Generate 3D View (Optional)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.arange(depth_map.shape[1]), np.arange(depth_map.shape[0]))
ax.plot_surface(X, Y, -depth_map, rstride=1, cstride=1, cmap='plasma', edgecolor='none')
plt.show()

# Display Results
print("Segmentation Map, Depth Map, and Suggested Design Variants Generated!")
