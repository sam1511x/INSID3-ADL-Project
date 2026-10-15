#!/usr/bin/env bash
# Download / prepare evaluation datasets into ./data (commands from docs/data.md).
# Usage: bash scripts/download_data.sh coco spair isic lung suim permis isaid pascal_part lvis paco_part
#        bash scripts/download_data.sh all
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data && cd data
pip install -q gdown==5.2.0 pandas >/dev/null

coco() {
  mkdir -p COCO2014 && cd COCO2014
  [ -d train2014 ] || { wget -q --show-progress http://images.cocodataset.org/zips/train2014.zip && unzip -q train2014.zip && rm -f train2014.zip; }
  [ -d val2014 ]   || { wget -q --show-progress http://images.cocodataset.org/zips/val2014.zip && unzip -q val2014.zip && rm -f val2014.zip; }
  if [ ! -d annotations/train2014 ]; then
    mkdir -p annotations && cd annotations
    gdown --fuzzy https://drive.google.com/file/d/1cwup51kcr4m7v9jO14ArpxKMA4O3-Uge/view
    gdown --fuzzy https://drive.google.com/file/d/1PNw4U3T2MhzAEBWGGgceXvYU3cZ7mJL1/view
    unzip -q train2014.zip && unzip -q val2014.zip && rm -f *.zip
    cd ..
  fi
  [ -d splits ] || { gdown --fuzzy https://drive.google.com/file/d/1K1UxhUzpjCe1jtgpsEUyrvU8gAlH1ZMy/view && unzip -q splits.zip && rm -f splits.zip; }
  cd ..
}

lvis() {
  mkdir -p LVIS/coco && cd LVIS/coco
  [ -d train2017 ] || { wget -q --show-progress http://images.cocodataset.org/zips/train2017.zip && unzip -q train2017.zip && rm -f train2017.zip; }
  [ -d val2017 ]   || { wget -q --show-progress http://images.cocodataset.org/zips/val2017.zip && unzip -q val2017.zip && rm -f val2017.zip; }
  cd ..
  [ -f lvis_val.pkl ] || { gdown --fuzzy https://drive.google.com/file/d/1itJC119ikrZyjHB9yienUPD0iqV12_9y/view && unzip -q lvis.zip && mv lvis/* . && rm -rf lvis lvis.zip; }
  cd ..
}

paco_part() {
  [ -d LVIS/coco/val2017 ] || lvis
  mkdir -p PACO-Part && cd PACO-Part
  ln -sfn ../LVIS/coco coco
  [ -d paco ] || { gdown --fuzzy https://drive.google.com/file/d/1VEXgHlYmPVMTVYd8RkT6-l8GGq0G9vHX/view && unzip -q paco.zip && rm -f paco.zip; }
  cd ..
}

pascal_part() {
  mkdir -p Pascal-Part && cd Pascal-Part
  if [ ! -d VOCdevkit/VOC2010/JPEGImages ]; then
    wget -q --show-progress http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
    tar -xf VOCtrainval_03-May-2010.tar && rm -f VOCtrainval_03-May-2010.tar
  fi
  if [ ! -f VOCdevkit/VOC2010/all_obj_part_to_image.json ]; then
    gdown --fuzzy https://drive.google.com/file/d/1WaM0VM6I9b3u3v3w-QzFLJI8d3NRumTK/view
    unzip -q pascal.zip && cp -r pascal/* VOCdevkit/VOC2010/ && rm -rf pascal.zip pascal
  fi
  cd ..
}

isic() {
  mkdir -p ISIC && cd ISIC
  if [ ! -d ISIC2018_Task1-2_Training_Input/1 ]; then
    wget -q --show-progress https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Training_Input.zip
    wget -q --show-progress https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Training_GroundTruth.zip
    unzip -q ISIC2018_Task1-2_Training_Input.zip && unzip -q ISIC2018_Task1_Training_GroundTruth.zip
    wget -q -O ISIC_Split.py https://raw.githubusercontent.com/Matt-Su/DR-Adapter/main/data_util/ISIC_Split.py
    mkdir -p isic && wget -q -O isic/class_id.csv https://raw.githubusercontent.com/Matt-Su/DR-Adapter/main/data_util/isic/class_id.csv
    sed -i "s|dir = '../data/ISIC/ISIC/ISIC2018_Task1-2_Training_Input/'|dir = 'ISIC2018_Task1-2_Training_Input/'|" ISIC_Split.py
    python ISIC_Split.py
    rm -f *.zip ISIC_Split.py && rm -rf isic && rm -f ISIC2018_Task1-2_Training_Input/*.jpg
  fi
  cd ..
}

lung() {
  mkdir -p LungSegmentation && cd LungSegmentation
  if [ ! -d CXR_png ]; then
    curl -sL -o chest-xray-masks-and-labels.zip https://www.kaggle.com/api/v1/datasets/download/nikhilpandey360/chest-xray-masks-and-labels
    unzip -q chest-xray-masks-and-labels.zip && mv 'Lung Segmentation'/* . && rm -rf 'Lung Segmentation' data *.zip
  fi
  cd ..
}

suim() {
  mkdir -p SUIM && cd SUIM
  if [ ! -d images ]; then
    curl -sL -o suim-merged.zip https://www.kaggle.com/api/v1/datasets/download/heyoujue/suim-merged
    unzip -q suim-merged.zip && mv suim_merged/* . && rm -rf suim_merged suim-merged.zip
  fi
  cd ..
}

isaid() {
  mkdir -p iSAID && cd iSAID
  if [ ! -d val/images ]; then
    gdown --fuzzy "https://drive.google.com/file/d/17PQ1iKCbaj2OjwBdCn_VBh09ntI4lxgL/view?usp=sharing" -O iSAID_5i.zip
    unzip -q iSAID_5i.zip && mv iSAID_patches/* .
    gdown --fuzzy https://drive.google.com/file/d/1RBmVHXGhEXak2XB4OeCzQj2yswOmuMB2/view
    unzip -q splits.zip && rm -rf *.zip iSAID_patches
  fi
  cd ..
}

permis() {
  # NOTE: needs ~45 GB of free disk temporarily (TAO test frames); ~0.3 GB after cleanup
  mkdir -p PerMIRS && cd PerMIRS
  if [ ! -d 0 ]; then
    wget -q --show-progress "https://motchallenge.net/data/3-TAO_TEST.zip"
    wget -q --show-progress https://omnomnom.vision.rwth-aachen.de/data/BURST/annotations.zip
    unzip -q 3-TAO_TEST.zip && unzip -q annotations.zip
    git clone -q https://github.com/dvirsamuel/PDM.git
    cd PDM/PerMIRS
    sed -i 's|base_path = f"/PerMIRS/{i}"|base_path = f"../../{i}"|' permirs_gen_dataset.py
    PYTHONPATH=.:.. python permirs_gen_dataset.py --images_base_dir ../../frames --annotations_file ../../test/all_classes.json
    cd ../..
    rm -rf PDM *.zip test train val frames info
  fi
  cd ..
}

spair() {
  mkdir -p SPair-71k && cd SPair-71k
  if [ ! -d JPEGImages ]; then
    wget -q --show-progress https://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz
    tar -xzf SPair-71k.tar.gz --strip-components=1 && rm -f SPair-71k.tar.gz
  fi
  cd ..
}

ALL="coco spair isic lung suim permis isaid pascal_part lvis paco_part"
[ "${1:-}" = "all" ] && set -- $ALL
for ds in "$@"; do
  echo "=== preparing $ds ==="
  $ds
done
echo "done."
