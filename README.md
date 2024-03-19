# TIS-camera-loader-for-YOLOv5
Up to 4 TIS (The Imaging Source) cameras can be used. Images will be concatenated and fed to model.

Add following statement somewhere before detection loop:
    dataset = LoadV4TISCams(source, img_size=640, stride=32, auto=True)

You can get image with following statement:
    for source, frame_lb, frame, rbt_flag, bad in dataset:

