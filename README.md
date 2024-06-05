# Vision Transformers Workshop
Welcome to the Vision Transformer Workshop! This repository contains materials, code, and resources for a comprehensive workshop designed to introduce the powerful capabilities of Transformer models. Whether you're dealing with seismic data or remote sensing data, this workshop aims to equip you with the knowledge and tools to leverage Transformers in your research and projects.

## Environment Setup Instructions

Please install a Miniconda environment. You will find a Windows, macOS or Linux version here: https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html

- Clone this repository by using this line of code: 
        
        git clone https://github.com/Adeniyilowee/Vision-Transformer-Workshop.git

- Open a terminal and change directory to the git cloned repository path:

        cd Vision-Transformer-Workshop

- Then update conda:

        conda update -n base -c defaults conda

- Type "yes" or "y" and press enter

- Then create a new conda environment using:

        conda env create -f environment.yml

- Activate the environment: 

        conda activate vit_workshop_2024

- Manually install torch vision using: 

        conda install torchvision -c pytorch

- Type "yes" or "y" and press enter to install all the required initialization dependencies.

- And then execute this to make the environment visible: 

        python -m ipykernel install --user --name vit_workshop_2024 --display-name "vit_Workshop_2024"
- Finally to make all files and folder sync:

        pip install -e .

- Run: `jupyterlab` At this point your default browser should open a page where you will see the folder with all the git repository files.

After the workshop, you can deactivate your conda environment via: `conda deactivate`

## Acknowledgments

Many thanks to the following institutions that made this work possible:

- German Climate Computing Center (DKRZ), Hamburg, Germany
- Helmholtz Center Hereon, Geesthacht, Germany
- Helmholtz AI

This work was supported by Helmholtz Association's Initiative and Networking Fund through Helmholtz AI [grant number: ZT-I-PF-5-01]. 
I also used resources of the Deutsches Klimarechenzentrum (DKRZ) granted by its Scientific Steering Committee (WLA) under project ID AIM.