## Traffic Light Detector

### Done List
- SSD detector that produces bounding boxed images

`ssd_mobilenet_v1_coco` detector performs very well on simulator images, but very bad on test site images. This leads to two approaches.

1. Use a lower confidence threshold gives better performance on test site images. Or do some data augmentation and cropping to get a better result. Then take the patches to train a classifier. This requires us to construct a training data. We might need to manually label the data.
2. Transfer learning on the `ssd_mobilenet_v1_coco` to get a better result on test site images. We need to manually draw the bounding boxes and label (red or green) the area. The learned detector will be able to recognize red and green light directly.

### To-do List
- classifier to classify red and green lights, it takes patches
- transfer learning `ssd_mobilenet_v1_coco` to get a one-shot detector
