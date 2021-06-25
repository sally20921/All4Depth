# Reproduction of SOTA Monocular Depth Estimation Methods
```
wget -i splits/kitti_archieves_to_download.txt -P kitti_data/
cd kitti_data/
unzip "*.zip"
cd ..
find kitti_data/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2, 1x1, 1x1 {.}.png {.}.jpg && rm {}'
```
- The above conversion command creates images with default chroma subsampling `2x2, 1x1, 1x1`. 
