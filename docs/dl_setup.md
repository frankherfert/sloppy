# nvidia driver setup
`sudo add-apt-repository ppa:graphics-drivers/ppa`

`sudo apt install nvidia-driver-410`


Check if installation works: `nvidia-smi`

Run continously (in tmux pane): `watch -n1 nvidia-smi`

Purge if not working: `sudo apt-get purge nvidia-*`

# conda setup

latest version to bash install: `curl https://conda.ml | bash`

update: `conda update conda`

extensions: `conda install -c conda-forge jupyter_contrib_nbextensions`

# fastai
`conda create -n fastai_1 python=3.6`

`conda activate fastai_1`

install current pytorch

`conda install -c pytorch -c fastai fastai pytorch` 

`torchvision cudatoolkit-10`

`conda install jupyter`

verify installation

`python -c "import torch; print(torch.__version__)"`

`python -c "import torch; print(torch.cuda.device_count());"`