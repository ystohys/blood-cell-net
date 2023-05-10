# Setup

This code was built on a Macbook and tested on either Colab or Paperspace Gradient cloud GPUs servers which are based on Linux, so it would be ideal to run everything here using a Linux OS with GPU access. I highly recommend using conda (Anaconda or Miniconda) as your package manager as the code was built using a conda environment. 

Unfortunately, I will be unable to provide an environment.yml file for setup as the cloud GPU servers do not use conda as their package manager (their packages are already set up upon launch). Therefore, please install the following packages in a conda environment:

- torch
- torchvision
- pandas
- numpy
- scikit-learn
- matplotlib
- jupyterlab
- opencv-python (need to use pip install)
- pycocotools (need to use pip install)

Once everything is installed, activate the conda environment and use Terminal to navigate to the working directory containing this README file and the code. Together with the "Code Walkthrough" section of the report, the Jupyter notebook named "main.ipynb" will give a good introduction to how the code works!

If there are any issues with setting up or questions regarding the code, please feel free to contact me through my email [yousheng.toh@gmail.com].

# Data

Many thanks to Dr Aslan for providing this open-source dataset here: https://www.kaggle.com/datasets/draaslan/blood-cell-detection-dataset