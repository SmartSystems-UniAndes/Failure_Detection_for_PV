# Failure classification system of photovoltaic panels using RGB images and deep learning

## About

This repository contains the work developed by Espinosa, A, et al. in *Failure signature classification in solar photovoltaic plants using RGB images and convolutional neural networks*. Here is shown the CNN development for photovoltaic module detection with semantic segmentation (U-NET), binary failure classification (with and without), 
and quaternary failure classification (cracks, dust, shadows and no failure) implementations. [1]

## Work Environment

See [SETUP.md](SETUP.md).

## How it works?

Each of the [*semantic_segmentation*](semantic_segmentation), [*binary_classification*](binary_classification) and [*quaternary_classification*](quaternary_classification) folders, is a self content project for its specific purpose. So, for example in the *semantic_segmentation* folder, you can run the data agumentation process and the U-NET training if you want. Please note, that is important to create the folders which are requireds for the dataset for every script that may need them.

To run the U-NET training, first run the [*augmentation_function.py*](semantic_segmentation/preprocessing/augmentation_function.py) script to make the adata augmentation process, then, run the [*ss_u_net.py*](semantic_segmentation/u_net/ss_u_net.py) script to re train the model, and if you want to observe the filters that the net learned, run the [*load_features_filters.py*](semantic_segmentation/u_net/load_features_filters.py) script. For binary and quaternary classification, the process is the same.

The generated dataset for the transfer learn process, and the trained models are availabel in the following link: https://drive.google.com/drive/folders/1sUerXy1_MTf7PUm2DjsNVTz173euYAmJ?usp=sharing

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
@article{gaviria_machine_2022,
	title = {Machine learning in photovoltaic systems: A review},
	issn = {0960-1481},
	url = {https://www.sciencedirect.com/science/article/pii/S0960148122009454},
	doi = {10.1016/j.renene.2022.06.105},
	shorttitle = {Machine learning in photovoltaic systems},
	abstract = {This paper presents a review of up-to-date Machine Learning ({ML}) techniques applied to photovoltaic ({PV}) systems, with a special focus on deep learning. It examines the use of {ML} applied to control, islanding detection, management, fault detection and diagnosis, forecasting irradiance and power generation, sizing, and site adaptation in {PV} systems. The contribution of this work is three fold: first, we review more than 100 research articles, most of them from the last five years, that applied state-of-the-art {ML} techniques in {PV} systems; second, we review resources where researchers can find open data-sets, source code, and simulation environments that can be used to test {ML} algorithms; third, we provide a case study for each of one of the topics with open-source code and data to facilitate researchers interested in learning about these topics to introduce themselves to implementations of up-to-date {ML} techniques applied to {PV} systems. Also, we provide some directions, insights, and possibilities for future development.},
	journaltitle = {Renewable Energy},
	shortjournal = {Renewable Energy},
	author = {Gaviria, Jorge Felipe and Narváez, Gabriel and Guillen, Camilo and Giraldo, Luis Felipe and Bressan, Michael},
	urldate = {2022-07-03},
	date = {2022-07-01},
	langid = {english},
	keywords = {Deep learning, Machine learning, Neural networks, Photovoltaic systems, Reinforcement learning, Review},
	file = {ScienceDirect Snapshot:C\:\\Users\\jfgf1\\Zotero\\storage\\G96H46L2\\S0960148122009454.html:text/html},
},


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
[1] Jorge Felipe Gaviria, Gabriel Narváez, Camilo Guillen, Luis Felipe Giraldo, and Michael Bressan. Machine learning in photovoltaic systems: A review. ISSN 0960-1481. doi: 10.1016/j.renene.2022.06.105. URL https://www.sciencedirect.com/science/article/pii/S0960148122009454?via%3Dihub

[2] Espinosa, A. R., Bressan, M., & Giraldo, L. F. (2020). Failure signature classification in solar photovoltaic plants using RGB images and convolutional neural networks. Renewable Energy, 162, 249-256.


## Licenses

### Software
The software is licensed under an [MIT License](https://opensource.org/licenses/MIT). A copy of the license has been included in the repository and can be found [here](https://github.com/SmartSystems-UniAndes/PV_MPPT_Control_Based_on_Reinforcement_Learning/blob/main/LICENSE-MIT.txt).
