import cv2
import os
from collections import defaultdict
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class first_vid:
    def player_detection(
            self,
            results # tracked results from Video A
    ):

        out_dir = "output_A"
        video_out_path = os.path.join(out_dir, "video_A_annotated.avi") # output path for annotated video A

        # OpenCV setup for writing frames into video
        height, width = results[0].orig_img.shape[:2]
        fps = 24
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))


        # declaring a Dictionary to hold frame crops per player id
        self.player_crops = defaultdict(list)


        # loop for processing each frame, extracting frame crops for each player id and saving annotated video A
        for result in results:
            op_frame = result.orig_img.copy()
            boxes = result.boxes
            if boxes is None:
                out.write(op_frame)
                continue
            
            frame = result.orig_img
            for box in boxes:
                if int(box.cls) != 2: 
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id)

                crop = frame[y1:y2, x1:x2]

                self.player_crops[track_id].append(crop)

                cv2.rectangle(op_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(op_frame, f"ID {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            out.write(op_frame)
        out.release()
    

    def embedding_extraction(
            self,
            model, # ReID model
            transform # image transformations for ReID model
    ):
        player_gallery = {}

        with torch.no_grad():

            # loop for extracting embeddings for each player detected in Video A
            for pid, crops in self.player_crops.items():
                embeddings = []
                for crop in crops:
                    if crop.size == 0:
                        continue
                    img_tensor = transform(crop).unsqueeze(0).to(device)
                    emb = model(img_tensor).cpu().numpy().flatten()
                    embeddings.append(emb)

                # Average embedding across all frames for this player
                avg_embedding = np.mean(embeddings, axis=0)
                player_gallery[pid] = avg_embedding
        return player_gallery
    

class second_vid:
    def player_mapping(
            self,
            results, # tracked results from Video B
            player_gallery, # dictionary of embeddings from Video A
            model, # ReID model
            transform # image transformations for ReID model
    ):
        
        # output directory for annotated video B
        out_dir = "output_B"
        os.makedirs(out_dir, exist_ok=True)
        video_out_path = os.path.join(out_dir, "video_B_annotated.avi")

        # OpenCV setup for writing video
        height, width = results[0].orig_img.shape[:2]
        fps = 24  
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))
        

        # Start with ID maxed from Video A
        current_max_id = max(player_gallery.keys()) + 1
        
        # Mapping from Video B tracking ID to matched Video A global ID
        b_to_global_id = {}
        for result in results:
            frame = result.orig_img.copy()
            boxes = result.boxes
            if boxes is None:
                out.write(frame)
                continue
            
            for box in boxes:
                if box.cls is None or box.id is None:
                    continue
                
                if int(box.cls) != 2:  # Only process players
                    continue
                
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id)

                crop = frame[y1:y2, x1:x2]

                # Skip empty crops
                if crop.size == 0:
                    continue
                
                # Extract embedding
                img_tensor = transform(crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model(img_tensor).cpu().numpy()

                # Match with Video A embeddings
                similarities = []
                for ref_id, ref_emb in player_gallery.items():
                    score = cosine_similarity(emb, ref_emb.reshape(1, -1))[0][0]
                    similarities.append((ref_id, score))

                # Find best match
                best_match_id, best_score = max(similarities, key=lambda x: x[1])

                # Threshold for matching
                if best_score > 0.75:
                    global_id = best_match_id
                else:
                    global_id = current_max_id
                    current_max_id += 1

                b_to_global_id[track_id] = global_id

                # Annotate frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {global_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            out.write(frame)
        out.release()




