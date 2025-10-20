def remove_blank_lines(input_filepath, output_filepath):
    try:
        with open(input_filepath, "r", encoding="utf-8") as infile:
            lines = infile.readlines()

        non_blank_lines = [line for line in lines if line.strip()]

        with open(output_filepath, "w", encoding="utf-8") as outfile:
            outfile.writelines(non_blank_lines)

        print(
            f"Successfully removed blank lines from '{input_filepath}' and saved to '{output_filepath}'."
        )

    except FileNotFoundError:
        print(f"Error: The file '{input_filepath}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


import os

# Example usage - modify these paths as needed
# This script should be run from the preprocessing directory
# Use relative paths to access texts in the sibling 'texts' directory

if __name__ == "__main__":
    # Example: remove blank lines from a classical text file
    input_file = os.path.join("..", "texts", "classical-sanskrit", "bhagavata-purana.txt")
    output_file = os.path.join("..", "texts", "classical-sanskrit", "bhagavata-purana.txt")

    # Uncomment the line below to run the blank line remover
    # remove_blank_lines(input_file, output_file)

    print("Blank line remover utility ready. Update input_file and output_file paths, then uncomment the function call.")
