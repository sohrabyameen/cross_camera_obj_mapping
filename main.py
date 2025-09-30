from ultralytics import YOLO
import torch
import os
import torchreid
from torchvision import transforms

from utils import first_vid, second_vid


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = YOLO("best.pt")


# Tracking players from Video A and the Tracked co-ordinates are saved in results_1
first_video_path = "videos/tacticam.mp4"
results_1 = base_model.track(source=first_video_path, show=False, save=False, persist=True, tracker="botsort.yaml")



out_dir = "output_A"
os.makedirs(out_dir, exist_ok=True)  # output directory for annotated video A

first_vid_obj = first_vid()
first_vid_obj.player_detection(results_1) # Tracked results are passed to player_detection function from Utils.first_vid class


# Initialize ReID model for embedding extraction
model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=1000,
    pretrained=True
)
model.eval()
model.to(device)


# Define image transformations for ReID model
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)), 
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]    
    )
])

# Calling embedding_extraction function from Utils.first_vid class to extract embeddings for players detected in Video A 

player_gallery = first_vid_obj.embedding_extraction(model, transform)
# Player_gallery above is a dictionary with key as player id and value as the corresponding embedding from Video A


# Tracking players from Video B and the Tracked co-ordinates are saved in results_2
second_video_path = "videos/broadcast.mp4"
results_2 = base_model.track(source=second_video_path, show=False, save=False, persist=True, tracker="botsort.yaml")

# output directory for annotated video B
out_dir = "output_B"
os.makedirs(out_dir, exist_ok=True)

# Calling player_mapping function from Utils.second_vid class to map players detected in Video B to players detected in Video A using cosine similarity of embeddings
second_vid_obj = second_vid()
second_vid_obj.player_mapping(results_2,player_gallery, model, transform)




