
excluded_files=( )

# Get list of all notebooks in the same directory as this script
# and convert them to PDF
for f in ../*.ipynb; do
    # check if the file is not in the excluded list
    for excluded_file in "${excluded_files[@]}"; do
        if [[ "$f" == *"$excluded_file"* ]]; then
            continue 2
        fi
    done

    jupyter nbconvert --to pdf "$f" --output-dir export
done

# merge all PDFs into one
pdfunite export/*.pdf Erklärung\ Eigenleistung.pdf ../export.pdf

# remove the temporary directory
rm -rf export