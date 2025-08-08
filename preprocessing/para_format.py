def format_paragraphs(input_filepath, output_filepath):
    """
    Reads a text file, joins lines within each paragraph, and writes the
    formatted content to a new file. Paragraphs are separated by blank lines.

    Args:
        input_filepath (str): The path to the input text file.
        output_filepath (str): The path to the output text file.
    """
    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            content = infile.read()

        paragraphs = content.split('\n\n') # Split the content into paragraphs by double newlines
        formatted_paragraphs = []

        for para in paragraphs:
            # Remove leading/trailing whitespace from the paragraph
            para = para.strip()
            if para: # Only process non-empty paragraphs
                # Replace all newlines within the paragraph with a single space
                # Then split by whitespace and join with a single space to handle
                # multiple spaces or tabs between words.
                formatted_line = ' '.join(para.split())
                formatted_paragraphs.append(formatted_line)

        # Join the formatted paragraphs with double newlines for separation in the output file
        output_content = '\n\n'.join(formatted_paragraphs)

        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            outfile.write(output_content)

        print(f"Successfully formatted paragraphs from '{input_filepath}' to '{output_filepath}'")

    except FileNotFoundError:
        print(f"Error: The file '{input_filepath}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

input_file = 'atharva-paippalada-samhita.txt'
output_file = 'atharva-paippalada-samhita-formatted.txt'
format_paragraphs(input_file, output_file)
