import os
import json
import random

def create_pairs(image_dir, video_dir, output_json_path, num_cross_pairs):
    images = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

    same_person_pairs = []
    cross_person_pairs = []

    # Create same-person pairs
    for image in images:
        person_id = image.split('.')[0]
        corresponding_video = f"{person_id}.mp4"
        if corresponding_video in videos:
            same_person_pairs.append({
                "source": f"{image_dir}/{image}",
                "driving": f"{video_dir}/{corresponding_video}",
                "same_person": True
            })

    # Create cross-person pairs
    all_pairs = [(image, video) for image in images for video in videos if image.split('.')[0] != video.split('.')[0]]
    random.shuffle(all_pairs)
    for image, video in all_pairs[:num_cross_pairs]:
        cross_person_pairs.append({
            "source": f"{image_dir}/{image}",
            "driving": f"{video_dir}/{video}",
            "same_person": False
        })

    # Combine all pairs
    all_pairs = same_person_pairs + cross_person_pairs

    # Save to JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(all_pairs, json_file, indent=4)

    print(f"Created {len(same_person_pairs)} same-person pairs and {len(cross_person_pairs)} cross-person pairs.")
    print(f"Total pairs: {len(all_pairs)}")
    print(f"JSON saved to {output_json_path}")
    print("\nGenerated Pairs:\n")
    for pair in all_pairs:
        print(pair)

# Parameters
image_dir = '/content/drive/MyDrive/VASA-1-master/image'
video_dir = '/content/drive/MyDrive/VASA-1-master/video'
output_json_path = 'driving_video.json'

# adjust
num_cross_pairs = 6  

# Create pairs
create_pairs(image_dir, video_dir, output_json_path, num_cross_pairs)
