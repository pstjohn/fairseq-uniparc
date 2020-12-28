# fairseq-uniparc

## Install and setup

First, create a conda environment with 
`conda env create -f fairseq.yml`

I'm using [torch_scatter](https://github.com/rusty1s/pytorch_scatter), which doesn't seem to install directly with conda.
Then, install my fork of fairseq from [pstjohn/fairseq](https://github.com/pstjohn/fairseq) with `pip install --editable ./` in the resulting conda environment. 

On eagle, my environment is kept in `/projects/deepgreen/pstjohn/envs/fairseq`. Use `--prefix=...` during `conda env create` to specify a custom directory.
