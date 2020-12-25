# Getting Started

This document briefly describes how to install and use the code.


## Environment

We tested this code in the following environment:
 - Linux
 - Python 3
 - FFmpeg

Similar environments (e.g. with OSX, Python 2) might work with small modification, but not tested.


## Description

This is a python wrapper that opens a MPEG-2 video and extracts the DCT coefficients from an I-frame or the motion vectors from a P- or B-frame as a numpy array.


#### Install

 - Download FFmpeg (`git clone https://github.com/FFmpeg/FFmpeg.git`).
 - Go to FFmpeg home,  and `git checkout 864fdfa0627e21ee0b69e957c3413114185623a7`.
 - `make clean`
 - `patch -p1 < ../ffmpeg-864fdfa.patch`
 - `./configure --prefix=${FFMPEG_INSTALL_PATH} --enable-pic --disable-yasm --enable-shared`
 - `make`
 - `make install`
 - If needed, add `${FFMPEG_INSTALL_PATH}/lib/` to `$LD_LIBRARY_PATH`.
 - Modify `setup.py` to use your FFmpeg path (`${FFMPEG_INSTALL_PATH}`).
 - `./install.sh`


#### Usage

The python wrapper has three functions: `parse`, `get_num_gops`, and `get_num_frames`.

The following call parses the MPEG-2 raw data and returns the DCT coefficients from an I-frame or the motion vectors from a P- or B-frame in a numpy array.
```python
from mpeg2 import parse
parse([fname], gop_index=0, frame_index=0, representation_type=0)
```
 - __fname__: path to video (.mpg).
 - __gop\_index__: a GOP of the video.
 - __frame\_index__: a frame of the GOP.
 - __representation\_type__: `0` for DCT coefficients or `1` for motion vectors.
 
For example,
```
parse('input.mpg', 3, 8, 1)
```
returns the motion vectors of the 9th frame of the 4th GOP.

The following call returns the number of GOPs of a MPEG-2 video.
```python
from mpeg2 import get_num_gops
get_num_gops([fname])
```
 - __fname__: path to video (.mpg).

For example, 
```
get_num_gops('input.mpg')
```

The following call returns the number of frames of a MPEG-2 video.
```python
from mpeg2 import get_num_frames
get_num_frames([fname])
```
 - __fname__: path to video (.mpg).

For example, 
```
get_num_frames('input.mpg')
```
