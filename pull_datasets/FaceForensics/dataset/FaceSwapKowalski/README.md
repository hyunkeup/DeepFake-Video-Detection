# FaceSwap Image Sequence Manipulation

## Example video
![example video](../../images/faceswap.gif)

## Install

- Install python 2.7 and requirements file by running `pip install -r requirements.txt`
- Download the [dlib shape predictor landmarks file](http://kaldir.vc.in.tum.de:/FaceForensics/models/shape_predictor_68_face_landmarks.dat) and put it in this directory
- Clone the [FaceSwap github](https://github.com/MarekKowalski/FaceSwap) and copy the contents of the `FaceSwap` folder into our `FaceSwap/FaceSwap` folder and the file `candide.npz` into this folder (note that we need a `__init__.py` file in the FaceSwap folder)

## Usage

To manipulate a video run
```shell
python faceswap.py
    -i1 <source folder, images in format '04d.png'>
    -i2 <target folder, images in format '04d.png'>.
```
The default output will be in a new folder named `output`. We return the manipulated image sequence as well as the used mask. For more information run `python faecswap.py -h`.

Note: The FaceSwap script will manipulate `min(#frames of video1, #frames of video2)` frames as we always swap faces of two corresponding frames. This means that the number of manipulated frames for FaceSwap will be lower than our other two manipulation methods. 


## Requirements

- Python 2.7
- requirements.txt file

