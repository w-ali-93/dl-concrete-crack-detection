# Concrete Crack Detection Using Deep Learning Techniques

One Paragraph of project description goes here

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You will need to download the dataset from here:
https://digitalcommons.usu.edu/cgi/viewcontent.cgi?filename=2&article=1047&context=all_datasets&type=additional

And extract it to the root of the cloned repository.

Create a directory named 'inception_v3_history' in root.
Create a directory named 'simple_cnn_history' in root.
Create a directory named 'resnet50' in root.

### Installing

You will need the following Python packages installed in your environment:
numpy, pickle, keras, scikit-learn.

## Running the Scripts

1. Run 'parse_data.py' to create the appropriate train / test / validation splits.
2. Run 'simple_cnn.py' for Simple CNN experiment.
3. Run 'resnet50.py' for ResNet50 experiment.
4. First run 'inception_v3_opt.py' then 'load_history_inception_v3.py' to find optimal hyper-parameter value, and then run 'inception_v3.py' for InceptionV3 experiment.

## Authors

* **Syed Wajahat Ali** - *Initial work* - [w-ali-93](https://github.com/w-ali-93)
* **Kamil Marwat** - *Contributor*
* **Waqas Ahmad**  - *Contributor*

## Acknowledgments

* Wiggle, S., 2018. Concrete Crack Detection. https://github.com/Swig-DPI/Concrete_crack_detection
* Maguire, Marc; Dorafshan, Sattar; and Thomas, Robert J., "SDNET2018: A concrete crack image dataset for machine learning applications" (2018). Browse all Datasets. Paper 48. https://digitalcommons.usu.edu/all_datasets/48
* Chollet, F., et al., 2015. Keras. https://keras.io.
