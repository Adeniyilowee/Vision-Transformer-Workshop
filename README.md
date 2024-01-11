# Understanding Transformers Workshop
Hello! Thank you for your interest in our vision transformer workshop!

### Environment Setup Instructions

Please install a Miniconda environment. You will find a Windows, macOS or Linux version here: https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html

Open a terminal and change directory to the git repository path: `cd cloned_repository_path`

We will now create a new conda environment with Python 3.8.10 as follows:

`conda create --name transformer_workshop python=3.8.10`

Type "yes" or "y" and press enter to install the required initialization dependencies.

We will activate the environment: 
`conda activate transformer_workshop`

And finally, we will install all the dependencies we need, in this case only pytorch:

`conda install -c pytorch pytorch==1.13.0`

Next, please install the Jupyter notebook: `pip install notebook`

Set up the newly created conda environment as a jupyter kernel:
- First install the necessary dependency: `pip install ipykernel`
- And then execute: 

`python -m ipykernel install --user --name transformer_workshop --display-name "TransformerWorkshop"`

Run the jupyter notebook start command in the terminal: `jupyter notebook` At this point your default browser should open a page where you will see the folder with all the git repository files. Double click on it.

A page will open with the code for mini-gpt. Select the correct kernel name (i.e. "TransformerWorkshop") from the dropdown list on the right side of the page.

If everything was setup up correctly in the previous steps, you should be able to run the import statements in the first notebook cell and not get any dependency errors. Note that running a cell means we select the cell and click the triangle/"play" button, however your browser may offer other convenient keyboard shortcuts for this.

If imports are fine, run subsequently all the other cells to see if you don't get any runtime errors.

To terminate the jupyter notebook session press Ctrl+C and select "yes"/"y" option.

After the workshop, you can deactivate your conda environment via: `conda deactivate`