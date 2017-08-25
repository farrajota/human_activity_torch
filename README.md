# Human Activity Recognition using Torch7

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

Recognize human activities of individuals using body pose joint annotations on video sequences.


## Installation

### Requirements

- NVIDIA GPU with compute capability 3.5+ (2GB+ ram)
- [Torch7](http://torch.ch/docs/getting-started.html)
- [dbcollection](https://github.com/dbcollection/dbcollection-torch7)

### Packages/dependencies installation

To use this example code, some packages are required for it to work.

```bash
luarocks install loadcaffe
luarocks install cudnn
luarocks install display
```

### dbcollection

To install the dbcollection package do the following:

- install the Python module (Python>=2.7 and >=3.5).

```
pip install dbcollection
```

- install the Lua/Torch7 dbcollection wrapper:

    1. download the Lua/Torch7 git repo to disk.

    ```
    git clone https://github.com/dbcollection/dbcollection-torch7
    ```

    2. install the package.
    ```
    cd dbcollection-torch7 && luarocks make
    ```

> For more information about the dbcollection package see [here](https://github.com/dbcollection/dbcollection-torch7).


# Getting started

## Download/Setting up this repo

To start using this repo you'll need to clone this repo into your home directory:

```
git clone https://github.com/farrajota/human_activity_torch
```

By default, this repo path points to `~/human_activity_torch/` when running the scripts. If you want to save/place this repo into another path/directory, you'll need to edit the `projectdir.lua` file in the repo's root dir and set the proper path to where you cloned it.

Next, the necessary data for this code to run is needed to be set. Download the `VGG16` model by running the `download_vgg.lua` script in the `download/` dir:

```
th download/download_vgg.lua
```

The `vgg16` model uses the `cudnn` library, so make sure this package is installed on your system before proceeding any further.


## Train

To train a network, simply run the `train.lua` script to start optimizing a pre-defined network (vgg16 + LSTM) on some default options. This script is configured to train the network specified in the paper [TODO- insert paper link]().

To train a network, there are several input arguments to configure the training process. The most important ones are the following (the rest you can leave as defaults):

- `-expID <exp_name>`: experiment id to store all the metadata, logs and model snapshots to disk under the `exp/` dir in the repo's main folder;
- `-dataset <dataset_name>`: indicates which dataset to train on (default=`ucf_sports`); **(warning: for now, the ucf_sports is the only available dataset to train/test on)**
- `-data_dir <path/to/dataset/files>`: Path to store the dataset\'s data files in case you haven't configured the `ucf_sports` previously in `dbcollection`. Specify a path if you want to store the data files into a specific folder.
- `-expDir <path/to/folder>`: specifies which folder to store the experiment directory. By default, it uses the `exp/` dir in the repo's main directory. (optional)
- `-netType <net_type_name>`: specifies which network to train. Options: vgg16-lstm | hms-lstm | vgg16-hms-lstm | vgg16-convnet | hms-convnet | vgg16-hms-convnet.
- `-trainIters <num_iters>`: Number of train iterations per epoch (default=300).
- `-testIters <num_iters>`: Number of test iterations per epoch (default=100).
- `-nEpochs <num_epochs>`: Total number of epochs to runh (default=15).
- `-batchSize <size>`: Mini-batch size (default=4).
- `-seq_length <size>`: Sequence length (number of frames per window) (default=10).

For more information about the available options, see the `options.lua` file or type `th train.lua -help`

> Note: In the `scripts/` dir there are several scripts to train + test other networks with different architectures and configurations. Check them out to see more training configurations specs.

## Test

Evaluating a network is done by running the `test.lua` script with some input arguments. To do so, you need to provide the following input arguments:

- `-expID <exp_name>`: experiment id that contains the model's snapshots + logs;
- `-loadModel <path/to/network.t7>`: if this flag is used, it bypasses the `-expID` flag and loads a specific file;
- `-dataset <dataset_name>`: indicates which dataset to test on (default=`ucf_sports`);
- `-test_progressbar false`: displays text information per iteration instead of a progress bar (optional);
- `-test_load_best false`: if true, it loads the best accuracy model (if exists, optional).

The results of the test, when finished, are displayed on screen and stored in the folder of the network file with the name `Evaluation_full.log`.

## Demo

To see the activity predictor model in action, the available demo displays the full sequence of images of a video/s in the browser. To launch the demo, you need to run `demo.lua` with some input arguments to specify which network to use and some other options:

- `-expID <exp_name>`: experiment id that contains the model's snapshots + logs;
- `-loadModel <path/to/network.t7>`: if this flag is used, it bypasses the `-expID` flag and loads a specific file;
- `-dataset <dataset_name>`: indicates which dataset to test on (default=`ucf_sports`);
- `-demo_nvideos <num_videos>`: number of samples to display predictions (default=5);
- `-demo_video_ids {id1, id2, id3, ..., idn}`: selects specific video ids to display (disables `-demo_nvideos` option);
- `-demo_topk <num>`: Display the top-k activities.
- `-demo_plot_save false`: save the image plots to disk into the `results/` dir (if true).

> Note: Before running this script, you must be sure you have started a display server on your local machine. You can start it by simply running `th -ldisplay.start` on a separate shell. Then, open up a tab on your browser and go to `http://localhost:8000/` to visualize the results.
(For more information about the `display` package and how to set up tthe server with different configurations go to [https://github.com/szym/display](https://github.com/szym/display))

## Results

TODO

## License

MIT license (see the [LICENSE](LICENSE.md) file)