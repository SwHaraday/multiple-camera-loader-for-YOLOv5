# multiple-camera-loader-for-YOLOv5
## Up to 4 TIS (The Imaging Source) cameras can be used. Images will be concatenated and fed to model.
### Also 4 Webcams can be used.
    - class LoadT4TISCams for tiled TIS camera images
    - class LoadV4TISCams for vertically concatenated TIS camera images
    - class LoadT4Streams for tiled Webcam images
    - class LoadV4Streams for vertically concatenated Webcam images

### Add following statement somewhere before detection loop:
    dataset = LoadV4TISCams(source, img_size=640, stride=32, auto=True)

### You can get image with following statement of detection loop:
    for source, frame_lb, frame, rbt_flag, bad in dataset:

#### Tiled
![](https://github.com/SwHaraday/TIS-camera-loader-for-YOLOv5/blob/main/sample_image/tiled.jpg)
#### Vertical
![](https://github.com/SwHaraday/TIS-camera-loader-for-YOLOv5/blob/main/sample_image/vertical.jpg)

