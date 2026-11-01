
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import os
import random
import kagglehub

# Download and extract dataset using kagglehub
path = kagglehub.dataset_download("iamsouravbanerjee/animal-image-dataset-90-different-animals")
path = os.path.join(path, "animals")
print(path)

def quick_clip_test(dataset_path, QuickTest=True, num_images=20):

    print("Scanning dataset folders...")
    #Mapping from species to list of image paths
    species_to_images = {}
    #List of all image paths
    all_image_paths = []
    #Go through dataset directory
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(root, file)

                # species folder is the immediate folder name
                species = os.path.basename(os.path.dirname(full_path))
                # add to mapping if not already present
                if species not in species_to_images:
                    species_to_images[species] = []
                # add image path to species list
                species_to_images[species].append(full_path)
                all_image_paths.append(full_path)
    # List of all species
    species_list = list(species_to_images.keys())

    print(f"Total species: {len(species_list)}")
    print(f"Total images found: {len(all_image_paths)}")

    #Mode of testing
    if QuickTest:
        images = random.sample(all_image_paths, min(num_images, len(all_image_paths)))
    else:
        images = all_image_paths
        num_images = len(images)

    print(f"Images selected for evaluation: {len(images)}")

    
    # initialize CLIP model and processor
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)

    prompts = [f"a photo of a {specie}" for specie in species_list]

    # Start Testing loop
    correct = 0
    incorrect = 0
    mistakes = []
    wrong_predictions = {}
    total_processed = 0

    print("\nRunning CLIP evaluation...\n")

    for image_path in images:
        print("loading image:", image_path  )
        try:
            # Species from folder name
            folder_species = os.path.basename(os.path.dirname(image_path))
            # print(f"Processing image: {image_path} (Species: {folder_species})")
            true_animal = folder_species

            # Load image
            image = Image.open(image_path).convert("RGB")
            inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)

            with torch.no_grad():
                outputs = model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)
                top_prob, top_idx = torch.max(probs, dim=1)

            predicted_animal = species_list[top_idx.item()]
            confidence = top_prob.item()

            is_correct = (predicted_animal == true_animal)

            # track stats
            if is_correct:
                correct += 1
            else:
                incorrect += 1
                mistakes.append((predicted_animal, true_animal, confidence))
                wrong_predictions[(predicted_animal, true_animal)] = (
                    wrong_predictions.get((predicted_animal, true_animal), 0) + 1
                )

            total_processed += 1

        except Exception as e:
            print(f"Error loading {image_path}: {e}")

    
    print("\n====== FINAL RESULTS ======")
    if total_processed > 0:
        accuracy = correct / total_processed
    print(f"\nFINAL RESULTS:")
    print(f"ACCURACY: {accuracy:.2%} ({correct}/{total_processed} correct)")
    print(f"Images processed: {total_processed}")
    print(f"Images skipped: {num_images - total_processed}")
    if incorrect > 0:
        print(f"Incorrect predictions: {incorrect}")
        print("Top 10 mistakes:")
        sorted_wrong_predictions = sorted(
            wrong_predictions.items(), 
            #Testing the counts instead of the keys
            key=lambda x: x[1], 
            reverse=True
        )
        for (predicted, true), count in sorted_wrong_predictions[:10]:
            print(f"Predicted: {predicted:<15} Actuals: {true:<15} MistakesCounts: {count:<10}")
    else:
        print("No valid images processed!")


# Run quick test
quick_clip_test(path,False)