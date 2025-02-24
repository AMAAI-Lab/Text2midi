import os
import json
import numpy as np
import librosa
import torch
from transformers import ClapModel, AutoProcessor

def load_captions_with_paths(caption_file):
    """Load captions and corresponding WAV file paths from JSON lines."""
    with open(caption_file, 'r') as file:
        return [json.loads(line) for line in file]

def compute_similarity_scores(captions_data, model, processor, output_file, chunk_duration=20, sr=48000):
    """Compute similarity scores for each WAV file and caption."""
    total_scores = []
    num_chunks = 0

    with open(output_file, 'w') as file:
        for item in captions_data:
            wav_filepath = item.get("location")
            caption = item.get("caption")
            if not os.path.isfile(wav_filepath):
                print(f"Missing WAV file: {wav_filepath}")
                continue

            print(f"Processing: wave file: {wav_filepath}, text prompt: {caption}")
            try:
                # Load the audio file
                audio_sample, _ = librosa.load(wav_filepath, sr=sr, mono=True)

                # Preprocess the text input
                text_input = processor(text=[caption], return_tensors="pt")

                # Generate text embeddings
                text_embeddings = model.get_text_features(**text_input)
                text_embeddings = text_embeddings / torch.norm(text_embeddings, dim=1, keepdim=True)

                # Split the audio into chunks
                chunk_length = int(chunk_duration * sr)
                chunks = [audio_sample[i:i + chunk_length] for i in range(0, len(audio_sample), chunk_length)]

                # Compute similarity scores for each chunk
                similarity_scores = []
                for chunk in chunks:
                    audio_input = processor(audios=chunk, return_tensors="pt", sampling_rate=sr)
                    audio_embeddings = model.get_audio_features(**audio_input)
                    audio_embeddings = audio_embeddings / torch.norm(audio_embeddings, dim=1, keepdim=True)

                    similarity_score = torch.mm(text_embeddings, audio_embeddings.T).item()
                    if similarity_score >= 0.01:  # Filter low scores
                        similarity_scores.append(similarity_score)

                # Add filtered chunk scores to the total scores list
                total_scores.extend(similarity_scores)
                num_chunks += len(similarity_scores)

                # Write scores for this file
                file.write(f"{wav_filepath}: {similarity_scores}\n")
                print(f"Written to file: {wav_filepath}")

            except Exception as e:
                print(f"Error processing {wav_filepath}: {e}")
                continue

    # Calculate the final average of filtered scores
    final_average = np.mean(total_scores) if num_chunks > 0 else 0
    with open(output_file, 'a') as file:
        file.write(f"\nFinal Average Similarity Score (Filtered): {final_average}\n")
    print(f"Final Average Similarity Score (Filtered): {final_average}")

# Example usage
caption_file = 'captions_wave_paths.txt'
output_file = "similarity_scores_with_average.txt"

# Load captions and file paths
captions_data = load_captions_with_paths(caption_file)

# Initialize the CLAP model and processor
model = ClapModel.from_pretrained("laion/larger_clap_music_and_speech")
processor = AutoProcessor.from_pretrained("laion/larger_clap_music_and_speech")

# Compute similarity scores
compute_similarity_scores(captions_data, model, processor, output_file)
