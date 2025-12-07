# JoshMeetsWorld - a C++, streaming capable fork of WORLD

JoshMeetsWorld is my fork of the WORLD repo.


My goal is to refactor the modules of WORLD into C++ classes that handle all their allocations in the constructor. It is very much a work in progress.


Please check the commit logs or reach out if you're curious about the state of things. :)


## What is WORLD?

WORLD was created by Masanori Morise and his many collaborators. I highly encourage you to read their publications on the algorithms contained in this repo!


Taken from the [WORLD repo](https://github.com/mmorise/World):
```
WORLD is free software for high-quality speech analysis, manipulation and synthesis.
It can estimate Fundamental frequency (F0), aperiodicity and spectral envelope and also generate the speech like input speech with only estimated parameters.

This source code is released under the modified-BSD license.
There is no patent in all algorithms in WORLD.
```

Some of the features in WORLD have also become quite popular for machine learning tasks. 


## Using the Repo

The repo is released under the same modified BSD-license as the original WORLD repo. You should consider doing the same with your modifications. :D  


## Building & Testing

### Windows
```
mkdir build
cd build

# the extra flag ensure tests are built
cmake .. -G "Visual Studio 17 2022" -DWORLD_BUILD_TESTS=true --fresh

cmake --build . --config Release

# Run Test Program
cd ..
.\build\Release\tests.exe .\test\vaiueo2d.wav vaieuotest_newd4cprocess.wav 1 1
```
