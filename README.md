# fairseq-uniparc

## Install and setup

First, create a conda environment with 
`conda env create -f fairseq.yml`

Then, install my fork of fairseq from [pstjohn/fairseq](https://github.com/pstjohn/fairseq) with `pip install --editable ./` in the resulting conda environment. 

On eagle, my environment is kept in `/projects/deepgreen/pstjohn/envs/fairseq`. Use `--prefix=...` during `conda env create` to specify a custom directory.
