#!/bin/bash

# Set output directory
OUTDIR=${1:-"./librispeech_data"}
mkdir -p "$OUTDIR"
cd "$OUTDIR" || exit 1

# List of LibriSpeech archives to download
FILES=(
  "train-clean-100.tar.gz"
  "train-clean-360.tar.gz"
  "train-other-500.tar.gz"
  "dev-clean.tar.gz"
  "dev-other.tar.gz"
  "test-clean.tar.gz"
  "test-other.tar.gz"
)

BASE_URL="https://openslr.trmal.net/resources/12"

echo "Downloading and extracting LibriSpeech data into $OUTDIR ..."

for FILE in "${FILES[@]}"; do
    echo "Downloading $FILE ..."
    wget -q --show-progress "$BASE_URL/$FILE" -O "$FILE"

    echo "Extracting $FILE ..."
    tar -xzf "$FILE"

    echo "Removing archive $FILE ..."
    rm "$FILE"
done

echo "✅ All datasets downloaded and extracted into $OUTDIR"
