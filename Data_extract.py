import os
import lzma
from tqdm import tqdm
import concurrent.futures
import random
import tempfile
import shutil
import logging
from datetime import datetime

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Set up logging to save files to the logs directory
log_filename = os.path.join(logs_dir, f'data_extract_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_filename
)

def process_file(args):
    directory, filename, temp_dir = args
    file_path = os.path.join(directory, filename)
    # Use a unique identifier (process ID + random number)
    unique_id = f"{os.getpid()}_{random.randint(1000, 9999)}"
    temp_output = os.path.join(temp_dir, f"{unique_id}_{filename}.txt")
    chars = set()
    
    try:
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            with open(temp_output, "w", encoding="utf-8") as outfile:
                # Process in chunks of 1MB to avoid memory issues
                chunk = infile.read(1024*1024)
                while chunk:
                    outfile.write(chunk)
                    # Update character set with each chunk
                    chars.update(set(chunk))
                    chunk = infile.read(1024*1024)
        
        return temp_output, chars
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        print(f"Error processing {file_path}: {e}")
        return None, set()

def xz_files_in_dir(directory):
    return [filename for filename in os.listdir(directory) if filename.endswith(".xz") and os.path.isfile(os.path.join(directory, filename))]

def process_files_in_parallel(files, folder_path, output_file, collect_vocab=False):
    success_count = 0
    failure_count = 0
    all_chars = set()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Process files and get temp file paths + character sets
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            args = [(folder_path, filename, temp_dir) for filename in files]
            for result in tqdm(executor.map(process_file, args), total=len(files)):
                results.append(result)
        
        # Merge temp files into output file
        with open(output_file, "a", encoding="utf-8") as outfile:
            for temp_path, chars in results:
                if temp_path and os.path.exists(temp_path):
                    success_count += 1
                    with open(temp_path, "r", encoding="utf-8") as infile:
                        shutil.copyfileobj(infile, outfile)
                    
                    # Collect vocabulary if requested
                    if collect_vocab:
                        all_chars.update(chars)
                else:
                    failure_count += 1
    
    logging.info(f"Processing complete: {success_count} files succeeded, {failure_count} files failed")
    print(f"Processing complete: {success_count} files succeeded, {failure_count} files failed")
    
    return all_chars if collect_vocab else None

def main():
    folder_path = "openwebtext"
    output_file_train = "output_train.txt"
    output_file_val = "output_val.txt"
    vocab_file = "vocab.txt"
    
    # Check if the source directory exists
    if not os.path.exists(folder_path):
        logging.error(f"Directory '{folder_path}' not found.")
        print(f"Directory '{folder_path}' not found.")
        return
    
    logging.info(f"Scanning for .xz files in {folder_path}")
    files = xz_files_in_dir(folder_path)
    total_files = len(files)
    logging.info(f"Found {total_files} .xz files")
    
    if total_files == 0:
        logging.error("No .xz files found in the directory.")
        print("No .xz files found in the directory.")
        return
    
    # Split into train and validation
    split_index = int(total_files * 0.9)  # 90% for training
    files_train = files[:split_index]
    files_val = files[split_index:]
    
    # Sample files (adjust sample_rate as needed)
    sample_rate = 0.01  # 1% of files
    files_train_sampled = random.sample(files_train, max(1, int(len(files_train) * sample_rate)))
    files_val_sampled = random.sample(files_val, max(1, int(len(files_val) * sample_rate)))
    
    logging.info(f"Processing {len(files_train_sampled)} training files and {len(files_val_sampled)} validation files")
    
    # Ensure output files are empty before appending
    for file_path in [output_file_train, output_file_val]:
        with open(file_path, 'w') as f:
            pass
    
    # Process files and collect vocabulary
    logging.info("Processing training files...")
    train_chars = process_files_in_parallel(files_train_sampled, folder_path, output_file_train, collect_vocab=True)
    
    logging.info("Processing validation files...")
    val_chars = process_files_in_parallel(files_val_sampled, folder_path, output_file_val, collect_vocab=True)
    
    # Combine vocabularies and write to file
    if train_chars or val_chars:
        all_chars = train_chars.union(val_chars) if train_chars and val_chars else train_chars or val_chars
        
        logging.info(f"Writing vocabulary with {len(all_chars)} unique characters to {vocab_file}")
        with open(vocab_file, "w", encoding="utf-8") as vfile:
            for char in sorted(all_chars):
                vfile.write(char + '\n')
    
    logging.info("Data extraction complete")
    print(f"Data extraction complete. Log saved to: {log_filename}")

if __name__ == "__main__":
    main()