#!/bin/bash
# Download MatsyaDx-BD v2 (CC BY 4.0) archives from Mendeley Data
mkdir -p data/matsyadx && cd data/matsyadx
B=https://data.mendeley.com/public-files/datasets/sxkynv9t7n/files
for pair in "efa8f617-6243-436d-8466-43a3f1f80006|Bacterial Gill Diseases.7z" "62c870a3-d382-406a-a1be-ba81e2ac239c|Epizootic Ulcerative Syndrome (EUS).7z" "11ea1958-3bd4-4478-9fed-3397cb561a7c|Bacterial Red Diseases.7z" "40b231a8-fc58-4f8e-ad8e-8fe02bcb633d|Healthy Fish.7z"; do
  id="${pair%%|*}"; name="${pair#*|}"
  curl -sL --retry 5 -C - -o "$name" "$B/$id/file_downloaded" && echo "done $name $(stat -c %s "$name")"
done
