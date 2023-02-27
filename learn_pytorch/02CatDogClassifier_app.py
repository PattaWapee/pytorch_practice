from fastapi import FastAPI, File, UploadFile
import torch
from torchvision import transforms
from PIL import Image


# Define the Pytorch model architecture
class CatDogClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(in_features=32*56*56, out_features=256)
        self.fc2 = torch.nn.Linear(in_features=256, out_features=2)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.ReLU()(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = torch.nn.ReLU()(x)
        x = self.pool(x)
        x = x.view(-1, 32*56*56)
        x = self.fc1(x)
        x = torch.nn.ReLU()(x)
        x = self.fc2(x)
        return x
        
# load model
model = CatDogClassifier() 
model.load_state_dict(torch.load('02inoutput/cat_dog_classifier.pth'))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
])



app = FastAPI()
@app.post('/predict')
def predict(file: UploadFile = File(...)):
    # read image
    image = Image.open(file.file)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
    ])
    image = transform(image)
    print(model)
    # make a prediction
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        predicted = torch.argmax(output).item()
        predict_class = 'dog' if predicted == 1 else 'cat'
    return {'prediction': predict_class}

# To run the app, go to the terminal and type:
# uvicorn 02CatDogClassifier_app:app --reload
# To test the app, go to the terminal and type:
# curl -X POST -F "file=@/Users/pattama/Desktop/Pattama/github/pytorch_practice/learn_pytorch/02inoutput/KOA_Nassau_2697x1517.jpg" http://localhost:8000/predict
