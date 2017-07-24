# Human Activity Recognition using Torch7

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

Recognize human activities of individuals using body pose joint annotations.

This work splits human activity recognition into three stages:

1. Detect persons/pedestrians using a [modified Fast R-CNN network](https://github.com/farrajota/pedestrian_detector_torch).
2. Detect human body joints using a [stacked auto-encoder network](https://github.com/farrajota/human_pose_estimation_torch) ([modified hourglass](https://github.com/anewell/pose-hg-train)).
3. Use a RNN to classify activities.

# WARNING

This repo is a work in progress. This warning shall be removed when code is working. For now, this repo represents only the scafold structure for the code.

## Installation

### Requirements

- NVIDIA GPU with compute capability 3.5+ (2GB+ ram)
- [Torch7](http://torch.ch/docs/getting-started.html)
- [dbcollection](https://github.com/dbcollection/dbcollection)

### Packages/dependencies installation

To use this example code, some packages are required for it to work.

```bash
luarocks install loadcaffe
```

### dbcollection

To install the dbcollection package do the following:

- install the Python module.

```
pip install dbcollection
```

- download the lua/torch7 git repository to disk.
```
git clone https://github.com/dbcollection/dbcollection-torch7
```

- install the Lua package.
```
cd dbcollection-torch7 && luarocks make
```

> For more information about the dbcollection package see [here](https://github.com/dbcollection/dbcollection).


# Getting started

## Download/Setting up this repo

TODO

## Train

TODO

## Test

TODO

## Demo

TODO

# License

MIT license (see the [LICENSE](LICENSE.md) file)