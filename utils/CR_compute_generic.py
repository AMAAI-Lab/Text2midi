import os
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor
import logging

# Set up logging
logging.basicConfig(filename='midi_process.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Function to find all .mid files in a directory
def find_midi_files(root_folder):
    midi_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(".mid"):
                midi_files.append(os.path.join(dirpath, filename))
    return midi_files

# Function to execute the command
def run_command(input_file, processed_files, error_files, jar_path, output_folder):
    command = ["java", "-jar", jar_path, "-i", input_file]
    
    if output_folder:
        command.extend(["-o", output_folder])

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        logging.info(f"Processed: {input_file}")
        processed_files.append(input_file)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error for {input_file}: {e.stderr}")
        error_files.append(input_file)

# Main function to handle arguments
def main():
    parser = argparse.ArgumentParser(description="Process MIDI files using Omnisia.")
    parser.add_argument("-i", "--input_folder", required=True, help="Path to the root folder containing MIDI files.")
    parser.add_argument("-o", "--output_folder", help="Path to the output folder (optional).")
    parser.add_argument("-j", "--jar_path", required=True, help="Path to the Omnisia JAR file.")
    
    args = parser.parse_args()

    # Find all MIDI files in the root folder
    input_files = find_midi_files(args.input_folder)

    processed_files = []
    error_files = []

    # Run commands in parallel
    with ThreadPoolExecutor() as executor:
        executor.map(run_command, input_files, [processed_files] * len(input_files), [error_files] * len(input_files), [args.jar_path] * len(input_files), [args.output_folder] * len(input_files))

    # Post-processing checks
    print(f"Total files processed successfully: {len(processed_files)}")
    print(f"Total files with errors: {len(error_files)}")

    if len(processed_files) + len(error_files) == len(input_files):
        print("All files were processed.")
    else:
        print("Some files were omitted.")
        missing_files = set(input_files) - set(processed_files) - set(error_files)
        print(f"Missing files: {missing_files}")

if __name__ == "__main__":
    main()
