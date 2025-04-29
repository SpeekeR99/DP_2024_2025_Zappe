import os
import sys
import re

# I was not really happy with the vlna programme by Petr Olšák, so I wrote my own
# It basically does the same thing, but additionally also adds ~ before \ref and after abbreviations and around --

if len(sys.argv) != 3:
    print(f"Usage: python {sys.argv[0]} <directory_containing_input_files> <directory_for_output_files>")
    sys.exit(1)

input_files = []
input_dir = sys.argv[1]
output_dir = sys.argv[2]

# Check if the input directory exists
if not os.path.exists(input_dir):
    print(f"Input directory '{input_dir}' does not exist.")
    sys.exit(1)

# Check if the output directory exists, if not create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get all .tex files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".tex"):
        print(f"Found input file: {filename}")
        input_files.append(os.path.join(input_dir, filename))

# Check if any .tex files were found
if not input_files:
    print(f"No .tex files found in the directory '{input_dir}'.")
    sys.exit(1)

# Process each input file
for input_file in input_files:
    print(f"Processing file: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Add ~ after every k, s, v, z, o, u and a, i
    content = re.sub(r"\s+(k|s|v|z|o|u|a|i)\s+", r" \1~", content)
    # If a whole paragraph is starting with K, S, V, Z, O, U, A, or I, keep the white spaces before those
    content = re.sub(r"(\n\s*)+(K|S|V|Z|O|U|A|I)\s+", r"\1\2~", content)
    # After the regex above, anything special already has ~ after it, so it won't be matched by this general one
    content = re.sub(r"\s+(K|S|V|Z|O|U|A|I)\s+", r" \1~", content)

    # Add ~ before every \ref
    content = re.sub(r"\s+\\ref", r"~\\ref", content)

    # Add ~ after every tj., tzv., tzn., např.
    content = re.sub(r"\s+(tj\.|tzv\.|tzn\.|např\.)\s+", r" \1~", content)

    # Add ~ around every --
    content = re.sub(r"\s+--\s+", r"~--~", content)

    # Done processing the file
    # Save the modified content to the output directory
    output_file = os.path.join(output_dir, os.path.basename(input_file))
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Processed file saved to: {output_file}")

# Print out the number of files processed
print(f"Processed {len(input_files)} files.")
