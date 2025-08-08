def remove_blank_lines(input_filepath, output_filepath):
    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()

        non_blank_lines = [line for line in lines if line.strip()]

        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            outfile.writelines(non_blank_lines)

        print(f"Successfully removed blank lines from '{input_filepath}' and saved to '{output_filepath}'.")

    except FileNotFoundError:
        print(f"Error: The file '{input_filepath}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

input_file = 'yajur-satapatha.txt'
output_file = 'yajur-satapatha.txt'

remove_blank_lines(input_file, output_file)
