from tinygrad import Tensor, nn
from tinygrad.nn.datasets import cifar
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad.dtype import dtypes

# CIFAR-10 class names
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Load CIFAR-10 dataset
X_train, Y_train, X_test, Y_test = cifar()

# Normalize test data (same as training)
X_test = X_test.float() / 255.0
Y_test = Y_test.cast(dtypes.int64)

# Define the same model architecture used during training
class SimpleCNN:
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.fc = nn.Linear(64 * 8 * 8, 10)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.conv1(x).relu().max_pool2d(2)
        x = self.conv2(x).relu().max_pool2d(2)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

# Create model and load weights
model = SimpleCNN()
state_dict = safe_load("cifar_net.safetensors")

# Map the param_X keys to the model parameters
from tinygrad.nn.state import get_parameters
params = get_parameters(model)

# Assign loaded weights to model parameters (move to same device)
for i, (k, v) in enumerate(state_dict.items()):
    params[i].assign(v.to(params[i].device))

print("Model loaded successfully from cifar_net.safetensors")

# Turn off gradient tracking for inference
Tensor.training = False

# Inference on test set with per-image printout
batch_size = 16  # smaller batch for easier printing
num_correct = 0
total_images = len(X_test)
print(f"Running inference on {total_images} test images...")

for i in range(0, len(X_test), batch_size):
    imgs = X_test[i:i+batch_size]
    labels = Y_test[i:i+batch_size]
    print(f"Processing batch {i//batch_size + 1}, images {i} to {min(i+batch_size, total_images)}")
    logits = model(imgs)
    preds = logits.argmax(axis=1)
    
    for j, pred in enumerate(preds):
        actual = int(labels[j].item())
        predicted = int(pred.item())
        print(f"Image {i+j}: Predicted = {classes[predicted]}, Actual = {classes[actual]}")
        if predicted == actual:
            num_correct += 1
    print(f"Batch complete, running accuracy: {num_correct}/{i+batch_size} = {num_correct/(i+batch_size):.4f}")

# Overall accuracy
accuracy = num_correct / len(X_test)
print(f"\nTest Accuracy: {accuracy:.4f}")
