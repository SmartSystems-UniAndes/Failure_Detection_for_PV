# Fault classification system of photovoltaic panels using RGB images and deep learning

## About

This repository contains the work developed by Andrés Florez in *Real time fault classification system of photovoltaic
panels using RGB images and deep learning*. Omitting the Raspberry Pi implementation, here is shown the CNN development
for photovoltaic module detection with semantic segmentation (U-NET), binary fault classification (fault and no fault), 
and quaternary fault classification (cracks, dust, shadows and no fault) implementations.

## Work Environment

See [SETUP.md](SETUP.md).

## Demo

To use the demo run the demo.py file that has the following arguments:

- **input_path**: Path where are allocated the demo images (default: *demo_images* folder)
- **output_path**: Path to save the segmented images (default: *demo_outputs* folder)
- **segmentation_model**: Path to the trained model for semantic segmentation (default: *.../models/UNet_segmentation_model
  .h5*).
- **bin_classification_model**: Path to the trained model for binary classification (default: *.../models/
  bin_classification_model.h5*).
- **quat_classification_model**: Path to the trained model for quaternary classification (default: *.../models/
  quat_classification_model.h5*).
- **mode**: Mode of use: *segmentation*, *bin_classification*, *quat_classification*, *all* (default: *all*).

Example:

```sh
$ python demo.py --mode all
```

## Citing Work

```BibTeX
@article{faultPVpanels,
  title = {Real time fault classification system of photovoltaic panels using RGB images and deep learning},
  author = {Andrés Florez, Luis Giraldo, Michael Bressan},
  journal = {} 
  year = {2021}
}
```