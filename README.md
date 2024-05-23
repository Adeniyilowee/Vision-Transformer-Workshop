# Vision Transformers Workshop
Hello! Thank you for your interest in our vision transformer workshop!

## Environment Setup Instructions

Please install a Miniconda environment. You will find a Windows, macOS or Linux version here: https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html

- Clone this repository by using this line of code: 

``` https://github.com/Adeniyilowee/Vision-Transformer-Workshop.git ```

- Open a terminal and change directory to the git cloned repository path:

`cd cloned_repository_path`

- Then create a new conda environment using:

.. code::
  conda env create -f environment.yml

- Type "yes" or "y" and press enter to install all the required initialization dependencies.

- Activate the environment: 

`conda activate vision_transformer_workshop`

Then manually install torch vision using: `conda install torchvision -c pytorch`

And then execute this to make the environment visible: `python -m ipykernel install --user --name vit_workshop --display-name "vit_Workshop"`

Run: `jupyterlab` At this point your default browser should open a page where you will see the folder with all the git repository files.

Try open one of the notebooks and run

To terminate the jupyter notebook session press Ctrl+C and select "yes"/"y" option.

After the workshop, you can deactivate your conda environment via: `conda deactivate`