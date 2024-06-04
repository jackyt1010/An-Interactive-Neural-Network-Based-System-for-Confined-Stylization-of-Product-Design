A localized NeuralStyleTransfer for Fashionable Product Image based on VGG-19(Initial Version)
======================

Neural Style Transfer is a task of transferring style
of one image to another. It is accomplished by using features of some pretrained model.
In this project, the employed model was used for localized stylization of specific shape
of the product image.GrabCut Algorithm and Region-Confined Stylization are being employed
for this research project.

## Table of content

- [System Requirements](#Requirement)
- [Installation](#Installation)
- [Run the Application](#Execution)
- [Usage](#Usage)
- [License](#license)
- [Reference](#Reference)

## Requirement
Operating system: Windows 10 <br />
Programming Langauge: Python 3 <br />
External Python Libraries: Please refer to the [requirements.txt](https://github.com/jackyt1010/An-Interactive-Neural-Network-Based-System-for-Confined-Stylization-of-Product-Design/blob/main/requirements.txt)

## Installation

```pip3 install -r requirements.txt```

## Execution
To run the application, please first download the [VGG-19 model](https://mega.nz/file/QDElFIZY#Gk99DLTosoDI-gvB8Fg6YvaBNDhVMyLptVJfeV_tDrY) into the main project directory and simply click the [run.bat](https://github.com/jackyt1010/An-Interactive-Neural-Network-Based-System-for-Contained-Stylization-of-Product-Design/blob/main/run.bat) to start.

## Usage
The following screenshot shows the completed execution result of the GUI of the Python application after clickling the run.bat
![](https://github.com/jackyt1010/An-Interactive-Neural-Network-Based-System-for-Contained-Stylization-of-Product-Design/blob/main/gui.jpg)

The sample of original images and style images are stored in the content folder and style folder respectively, the final stylized image is stored in the images folder as result.jpg

## License

This project is licensed under MIT License.

## Reference
* [An Interactive Neural Network-Based System for Confined Stylization of Product Design](https://github.com/jackyt1010/An-Interactive-Neural-Network-Based-System-for-Contained-Stylization-of-Product-Design/edit/main/README.md)

 Man-Kit, Tang, Fu-lai, Chung, Chun Yin Fan.”An Interactive Neural Network-Based System for Confined Stylization of Product Design”. In: Proceedings of International Conference on Design and Semantics of Form and Movement [(DesForM 2023](https://www.desform2023.org/)), p.107-109, June 2023.
* [Constrained Neural Style Transfer for Decorated Logo Generation](https://github.com/gttugsuu/Constrained-Neural-Style-Transfer-for-Decorated-Logo-Generation)
* [Grab Cut Algorithm Implementation](https://github.com/louisfb01/iterative-grabcut)
* [The Music used for the creation of the demo video](https://imperss.bandcamp.com/track/reflection)
