<h1 align="center">
  PyTorch Lightning Template Example <br> CoNLL 2003 <br> Named Entity Recognition
</h1>

<p align="center">
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-orange?style=for-the-badge&logo=pytorch"></a>
  <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-blueviolet?style=for-the-badge"></a>
  <a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra-blue?style=for-the-badge"></a>
  <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge"></a>
</p>
This repository contains an example based on the pytorch lightning bootstrap template from the <a href="https://github.com/edobobo/p-lightning-template">p-lightning-template</a> repository.
<br> We provide a minimalistic example of py-lighing module to address the NER task from CoNLL 2003.

### Dependencies
1. Conda
2. Linux or MacOS are recommended
3. unzip tool, check it executing ```which unzip```
4. git
## Dataset
The dataset used in the project is the Named Entity Recognition (NER) from the competition CoNLL 2003.<br>
The dataset is automatically downloaded from the pl_data_modules class and ready to be use.

## Using the repository
1. Clone the repository
```
git clone https://github.com/alescire94/p-lightning-template-example ner_conll03 && cd ner_conll03
```
2. Install all python3 dependencies, answering to setup.sh questions
The installer supports cuda for the GPU usage.
```
bash setup.sh
```
3. Execute the train script
```
PYTHONPATH=. python3 src/train.py
```
4. Test it
```
PYTHONPATH=. python3 src/scripts/evaluate.py 
```
4.a By default the evaluation script takes as input the sentence specified in conf/evaluate/default.evaluate.yaml 
To change it you can modify the file or by command line, as usual for all pytorch lightning parameters.
```
PYTHONPATH=. python3 src/scripts/evaluate.py -m evaluate.sentence="I am from Rome"
```


## FAQ
**Q**: When I run any script using a Hydra config I can see that relative paths do not work. Why?

**A**: Whenever you run a script that uses a Hydra config, Hydra will create a new working directory
(specified in the root.yaml file) for you. Every relative path you will use will start from it, and this is why you 
get the 'FileNotFound' error. However, using a different working directory for each of your experiments has a couple of 
benefits that you can read in the 
[Hydra documentation](https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/) for the Working 
directory. There are several ways that hydra offers as a workaround for this problem here we will report the two that
the authors of this repository use the most, but you can find the other on the link we previously mentioned:

1. You can use the 'hydra.utils.to_absolute_path' function that will convert every relative path starting from your 
working directory (p-lightning-template in this project) to a full path that will be accessible from inside the 
new working dir.
   
2. Hydra will provide you with a reference to the original working directory in your config files.
You can access it under the name of 'hydra:runtime.cwd'. So, for example, if your training dataset
has the relative path 'data/train.tsv' you can convert it to a full path by prepending the hydra 
variable before
   
# Authors
Alphabetic order equal contribution
* **Andrea Bacciu**  - [github](https://github.com/andreabac3)
* **Edoardo Barba**  - [github](https://github.com/edobobo)
* **Alessandro Scirè**  - [github](https://github.com/alescire94)
