<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="https://i.ibb.co/HdstZLw/final1.png" alt="Logo">
  </a>

  <h3 align="center">Link Prediction for Biomedical Data using 
Graph Neural Networks
</h3>

  <p align="center">
    Predicting disease-gene and disease-drug link using RGCN in DGL implementation
    <br />
  </p>
</p>


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This project explores Graph Convolutional Networks (GCN), specifically RGCN (Relational GCN) for graph embedding and link prediction of biomedical data relations: disease-gene and disease-drug.

Motivation:
* Learn relationships between entities in graphs, specifically biomedical data
* Learn graph neural networks
* Learn how to use Deep Graph Library and Pytorch

Datasets:
* [DG-Miner (disease-gene)](http://snap.stanford.edu/biodata/datasets/10020/10020-DG-Miner.html)
* [DCh-Miner (disease-drug)](http://snap.stanford.edu/biodata/datasets/10004/10004-DCh-Miner.html)

### Packages

To run this repository, you will need to install the following packages in your virtual Python environment:
* backports.lzma==0.0.14
* dgl-cu110==0.6.1
* imageio==2.9.0
* matplotlib==3.4.2
* numpy==1.20.2
* pandas==1.2.4
* pylzma==0.5.0
* scikit-learn==0.24.2
* scipy==1.6.3
* seaborn==0.11.1
* sklearn==0.0
* tensorboard-data-server==0.6.1
* tensorboard-plugin-wit==1.8.0
* tensorboard==2.5.0
* torch==1.8.1+cu111
* torchaudio==0.8.1
* torchsummary==1.5.1
* torchvision==0.9.1+cu111
* tqdm==4.61.0
* urllib3==1.26.4

<!-- GETTING STARTED -->
## Getting Started

Create a virtual environment with Python 3.6 or above and activate it.

### Installation

1. Install Pytorch. Please use this [Pytorch guide](https://pytorch.org/get-started/locally/) to get the right Pytorch version for your computer. For our lab machine, we use this version for CUDA 11.1.
   ```sh
	pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
   ```
2. Install DGL using this [DGL installation guide](https://www.dgl.ai/pages/start.html). For Linux:

	```sh
	pip install dgl
	```
3. Install other packages such as scikit-learn, tensorboard, pandas, etc.

4. Clone the repo
   ```sh
   git clone https://github.com/lerachel/css586-final.git
   ```
5. Download datasets to repo directory from [here](https://drive.google.com/drive/folders/1KcAMPcltQ_VdNRz06VVsol6cZVZQqwsa?usp=sharing). You must log in with UW email to be able to download. 

6. If step 5 is not applicable, you can download DG-Miner and DCh-Miner datasets from [BioSNAP website](http://snap.stanford.edu/biodata/index.html).


<!-- USAGE EXAMPLES -->
## Usage

Run Python files. For Rachel's example, to run model 7 with combined dataset, 2 edge type prediction, in_feat, hid_feat, out_feat, number of HeteroGraphCV:  
   ```sh
   python rachel_main.py --link 2 --in_feat 20 --hid_feat 20 --out_feat 20 --layer 2
   ```
To run rachel_disease_gene.py:
   ```sh
   python rachel_disease_gene.py
   ```
<br />
<p align="center">
  <a href="https://i.ibb.co/ZKc4P3y/Screen-Shot-2021-06-08-at-1-22-56-PM.png">
    <img src="https://i.ibb.co/ZKc4P3y/Screen-Shot-2021-06-08-at-1-22-56-PM.png" alt="run screenshot" width="407" height="427">
  </a>
  <p align="center">
    Example of Run screenshot for rachel_main.py
  </p>
</p>

<br />


<!-- CONTACT -->
## Contact

Rachel Le - lerachel@uw.edu

Chip Kirchner - ckirch@uw.edu

Project Link: [css586-final](https://github.com/lerachel/css586-final)

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Stanford Biomedical Network Dataset Collection](http://snap.stanford.edu/biodata/index.html)
* [Deep Graph Library](https://www.dgl.ai/)

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png