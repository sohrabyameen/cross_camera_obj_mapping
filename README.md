# cross_camera_obj_mapping

Player Re-Identification Across Multi-Camera Football Videos

This project focuses on tracking football players across two different camera angles using a fine-tuned object detection model and a deep ReID embedding-based similarity matching system. It assigns consistent player IDs across both videos, even when players appear from different perspectives.

Aim : To assign each football player a consistent and unique ID across two separate video clips recorded from different camera angles. This ID should remain the same even if the player:

Moves in and out of the frame,
Appears from different perspectives,
Changes apparent size due to zoom or camera position.


******** Methodology ********

1. Object Detection & Tracking
A fine-tuned YOLOv11 model is used to detect and track players in videos.
fine tuned model download link : https://drive.google.com/file/d/1ifnkUjKLvEQGU-rOg0h-qAYJRwMftsMv/view?usp=sharing


Tracking is done using BoT-SORT.


2. Feature Extraction with ReID
For video A , image crops of each player from each frame are used to extract embeddings.
Multiple image crops per player are averaged to create stable identity embeddings for each player.

A deep re-identification model (osnet_x1_0 from Torchreid) extracts robust visual embeddings for each frame crop of detected player.


3. Cross-Video ID Matching (Video B)
Same for Video B, embeddings are extracted from image crops for each player.

Embeddings of players in Video A are compared with Video B with the Cosine Similarity.

A global ID is assigned if similarity > 0.75; otherwise, a new ID is created.


******** Execution ********

Place both videos (tacticam.mp4, broadcast.mp4) inside the videos/ directory.

Make sure the pretrained YOLO model "best.pt" is in the main directory.

Install dependencies from requirements.txt

After executing the main.py:

Two new folders output_A/ and output_B/ will be created

Each contains an annotated .avi video with bounding boxes and global player IDs



******** Dependencies ********

Python ≥ 3.12
Ultralytics YOLOv8
TorchReID
OpenCV
PyTorch
Scikit-learn

You may need a CUDA-compatible GPU for faster inference (optional but recommended)



********** Code WorkFlow *************

utils.py

    Class first_vid

        function player_detection ( recives player tracking results from main.py )

            process each frame
            takes out a small crop of each player from each frame
            each frame gets annotated
            store multiple frame crops for each player in a dict player_crops
        
        function embedding_extraction( receives REID model and image transformation from main.py )

            for each player : extracts embeddings for each frame crop and average them
            store {player id : embedding} in a dict player_gallery
            returns player_gallery to main.py


    Class second_vid

        function player_mapping(
            vido B results from main.py,
            player_gallery from main.pu,
            model from main.py
            image transform for model from main.py
        ) :

            process each frame
            takes out a small crop of each player from each frame 
            for each player : extracts embeddings for each frame crop and average them
            compares them to player embeddings from Video A using cosine similarity
            A global ID is assigned if similarity > 0.75; otherwise, a new ID is created.
            frame is annotated 
            after all frames completed , video is released


main.py

    players in video A are tracked using base model

    results of video A are passed to utils.first_vid.player_detection

    REID model and image transformation function declared

    utils.first_vid.embedding_extraction is called to get embeddings

    players in video B are tracked using base model

    results of Video B, model, transform, embeddings from Video A are passed to utils.second_vid.player_mapping.



******** Scope of Improvement ********

1. Implementing player movement maps
2. Implementing deep identification of features like facial features, jersy color
3. A more robust and specifically fine tuned REID model


    Above techniques are not implemented due to limited resources and computing power






