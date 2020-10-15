## Code associated with:

### Word-level Loss Extensions for Neural Temporal Relation Classification
Artuur Leeuwenberg & Marie-Francine Moens
*In Proceedings of COLING*, Santa Fe, New Mexico, 2018.

#### Terms of Usage
The code may only be used for Academic purposes. In case of usage, please cite the corresponding paper. For commercial use please check the LICENSE.html.

### What this code can be used for:
Training and prediction of neural temporal relation classification models, with or without an additional word-level skip-gram objective for learning the model's word representations, i.e. to reproduce the results from the COLING 2018 paper. A tiny artificial dataset was added to run the code, as the THYME and MIMIC III require a license agreement so could not be released with the code. 


### How do I get set up? ###
Install  Python 2.7 (if not yet installed)

Install GraphViz  (if not present already) with:
```
sudo apt-get install graphviz
```

Setup and activate (line 2) a virtual environment, and install the Python dependencies with:
```
virtualenv venv -p python2.7
source venv/bin/activate
pip install -r requirements.txt
```

Install and setup the Stanford POS Tagger with Python NLTK with:
```
python -m nltk.downloader -d venv/nltk_data punkt
python -m nltk.downloader -d venv/nltk_data third-party
wget https://nlp.stanford.edu/software/stanford-postagger-2016-10-31.zip
unzip stanford-postagger-2016-10-31.zip
mv stanford-postagger-2016-10-31 stanford-postagger
rm stanford-postagger-2016-10-31.zip
mv english-caseless-left3words-distsim.tagger stanford-postagger/models/
mv english-caseless-left3words-distsim.tagger.props stanford-postagger/models/
```

### Running the code:
To train and predict with different models from the paper using the provided tiny artificial dataset (using CPU by default) run:
```
sh demo.sh
```
To get more information on how to use the code (e.g. use GPU) inspect `demo.sh` or run:
```
python run_experiment.py -h
```
### Reproducing the results from the paper:
To run the same models on the same data as those from the paper:

1. Obtain the [THYME](https://clear.colorado.edu/TemporalWiki/index.php/Main_Page) corpus and [MIMIC III](https://mimic.physionet.org/) dataset.
2. Place the THYME Train and Dev sections in `data/real/Train/`, and the THYME Test section in `data/real/Test/`, like the artificial data (corresponding .xml and .txt pairs in subfolders).
3. Place a copy of each raw .txt file from the THYME Train and Dev sections in `data/real/Raw/`
4. Obtain the used MIMIC III section from the MIMIC III `NOTEEVENTS_DATA_TABLE.csv` by running:
```
python get-mimic3-subsection.py NOTEEVENTS_DATA_TABLE.csv
```
and add the obtained .txt files from `mimic3-subsection-out/` to `data/real/Raw/`
5. The data should now be set up. You can run the script to train the models:
```
sh table2.sh
```

By default, the script runs on CPU. To run on GPU, change the `CUDA_VISIBLE_DEVICES` in `table2.sh`.

### Questions?
> Any questions? Feel free to send an email to tuur.leeuwenberg@cs.kuleuven.be
