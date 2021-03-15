# Failure classification system of photovoltaic panels using RGB images and deep learning

## About

This repository contains the work developed by Espinosa, A, et al. in *Failure signature classification in solar photovoltaic plants using RGB images and convolutional neural networks*. Here is shown the CNN development for photovoltaic module detection with semantic segmentation (U-NET), binary failure classification (with and without), 
and quaternary failure classification (cracks, dust, shadows and no failure) implementations. [1]

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
@article{espinosa2020failure,
  title={Failure signature classification in solar photovoltaic plants using RGB images and convolutional neural networks},
  author={Espinosa, Alejandro Rico and Bressan, Michael and Giraldo, Luis Felipe},
  journal={Renewable Energy},
  volume={162},
  pages={249--256},
  year={2020},
  publisher={Elsevier}
}
```

## References

[1] Espinosa, A. R., Bressan, M., & Giraldo, L. F. (2020). Failure signature classification in solar photovoltaic plants using RGB images and convolutional neural networks. Renewable Energy, 162, 249-256.
