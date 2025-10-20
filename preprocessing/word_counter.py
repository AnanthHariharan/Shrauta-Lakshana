import os


def count_words_in_directory(directory_path):
    """
    Calculates the total word count of all .txt files within a given directory
    and its subdirectories.
    """
    total_words = 0
    txt_files_processed = 0

    # os.walk traverses the directory tree
    for dirpath, _, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.lower().endswith(".txt"):
                filepath = os.path.join(dirpath, filename)

                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                        # A robust way to count words: split by whitespace
                        words = list(filter(None, content.split()))
                        total_words += len(words)
                        txt_files_processed += 1
                except Exception as e:
                    print(f"  [Error] Could not read file {filepath}: {e}")

    return total_words, txt_files_processed


def main():
    # Define the path to the parent 'texts' folder.
    # Since the script is in 'preprocessing', '..' moves up one level,
    # and then 'texts' points to the sibling folder.
    TEXTS_ROOT = os.path.join("..", "texts")

    # The specific folders to analyze within the TEXTS_ROOT
    target_folders = ["samhita", "brahmana", "upanishad", "classical-sanskrit"]

    print(f"--- Starting Text Analysis from {TEXTS_ROOT} ---")

    # Check if the root texts directory exists
    if not os.path.isdir(TEXTS_ROOT):
        print(f"\nERROR: The directory '{TEXTS_ROOT}' was not found.")
        print(
            "Please ensure the folder structure is correct (word_counter.py and texts are siblings)."
        )
        return

    grand_total_words = 0

    for folder_name in target_folders:
        folder_path = os.path.join(TEXTS_ROOT, folder_name)

        if not os.path.isdir(folder_path):
            print(f"[{folder_name}] Folder not found. Skipping...")
            continue

        # Calculate word count for the specific subfolder
        word_count, file_count = count_words_in_directory(folder_path)
        grand_total_words += word_count

        print(f"\n=======================================================")
        print(f"  Folder: {folder_name.upper()}")
        print(f"  Files Analyzed: {file_count} .txt files")
        print(f"  Total Word Count: {word_count:,}")
        print(f"=======================================================")

    print(
        f"\n--- GRAND TOTAL ACROSS ALL TARGET FOLDERS: {grand_total_words:,} words ---"
    )


if __name__ == "__main__":
    main()
