# fastms

This code implements the approach for the following research paper:

> **Real-Time Minimization of the Piecewise Smooth Mumford-Shah Functional**, *E. Strekalovskiy, D. Cremers*, European Conference on Computer Vision (ECCV), 2014. ([pdf](https://vision.in.tum.de/_media/spezial/bib/strekalovskiy_cremers_eccv14.pdf)) ([video](https://vision.in.tum.de/_media/spezial/bib/strekalovskiy_cremers_eccv14.mp4))

![alt tag](https://vision.in.tum.de/_media/data/software/fastms.png)

The algorithms computes piecewise smooth and piecewise constant approximations to input images. The result will be smooth over more or less large regions, but is allowed to have sharp color jumps (discontinuities) between regions of smoothness. Applications range from image denoising to producing cartooning image effects. The model describning how exactly the result should look like -- the Mumford-Shah model -- has a long history and is among the most cited in computer vision.
Current state-of-the-art approaches either use heuristics introducing additional artificial parameters, which leads to parameter-sensitive results, or require minutes of run time for a single VGA color image, see the above paper for more details.

The approach implemented here produces state-of-the-art quality results and runs in real-time.

## Features

- **GPU implementation** using CUDA, and a **CPU implementation** optionally using OpenMP. Either implementation can be chosen using a command line parameter without recompiling the code.
- **float or double** precision. 
- **MATLAB wrapper** for quick prototyping.


# 1. Installation

## 1.1 Quick start

Install:

        git clone https://github.com/tum-vision/fastms.git

        cd fastms

        make

Run:

        ./main

Or run any of the examples, e.g.

        ./examples/example1.sh

## 1.2 Requirements

### CUDA:

To use the **GPU implementation**, [CUDA](https://developer.nvidia.com/cuda-downloads) must be installed and *nvcc* must be in the current PATH.
The code is generated for NVIDIA GPUs of [compute capability](https://developer.nvidia.com/cuda-gpus) at least 2.0 by default.
If your GPU is older, uncomment the corresponding *gencode* line in the *Makefile*.

*Note: You can still compile and use the* **_CPU version_**, *even if CUDA is not available.*

### OpenCV

The actual algorithm implementation is *independent* of OpenCV.

[OpenCV](http://opencv.org/downloads.html) is only used for the example usages of the algorithm (*./main*), to read/write and display the images. So you should have it installed for a quick start, but you can of course use any other library of your choice for image reading/writing/displaying, e.g. [Qt](http://qt-project.org).

### MATLAB

If you want to use the provided **MATLAB wrapper**, MATLAB must be installed, of course.


#### Notes:

The code was tested on two different systems:

- Ubuntu 12.04 (Precise) with CUDA 5.5 and OpenCV 2.3.1,
- Mac OS X 10.9 (Mavericks) and 10.10 (Yosemite) with CUDA 6.5 and OpenCV 2.4.8.
- - the CPU-only version (disable *USE_CUDA* in the *Makefile*) can be compiled with either *clang* or *gcc* of any version.
- - the CUDA version compiled only with *gcc* on our test system, namely with *gcc-4.2* (installed through [homebrew](http://brew.sh)). Note that OpenCV needs to be compiled with the same version of *gcc*.



# 2 Usage

Run the code using 

        ./main -i <your_input_images>

See Section 4 for all possible parameters.

## 2.1 GUI with live results

If you specify only one image, e.g.

        ./main -i images/hepburn.png

you will get a GUI where one can change the main regularization parameters *lambda* and *alpha of the Mumford-Shah model, and see the results immediatelly.

### Hotkeys
- <kbd>e</kbd>: Toggle edges highlighting on/off
- <kbd>v</kbd>: Toggle info printing on/off
- <kbd>w</kbd>: Toggle regularizer weighting on/off
- <kbd>s</kbd>: Save current result image
- <kbd>ESC</kbd>: Exit

## 2.2 Batch processing

You can provide more than one image if you like, e.g.

        ./main -i images/hepburn.png -i images/gaudi.png

        ./main -i images/*.png

In this case, the given parameters will be applied to all images one after another, optionally with temporal regularization (if "-temporal <float>" is specified, see Section 4 below), the resulting images will be saved, and then displayed one after another.
Press any key to switch to the next image.

## 2.3 Live camera stream

Run

        ./main -cam

(and optionally some other parameters) to process the frames from a plugged-in **USB camera** (or the built-in camera) in real-time.
A GUI will be shown with the same active *hotkeys as in Section 2.1*.

For example, press **'s'** to save the result for the current frame.

*Note: This requires that the camera is recognizable using OpenCV. This worked fine on Linux, but didn't work on Mac OS X on our test system.*

## 2.4 Running the examples

### On images

For a quick start we provide some sample input images in the "images" directory.
The directory *./examples* contains some example usages of the command line tool.
Run e.g.

        ./examples/example1.sh

        ./examples/example2.sh

etc.

### On video

There is also the *video* "./images/video.mp4". To run the code on the video:

- first *extract the video frames*, by running (inside the "images" directory)

        ./video_extract_frames.sh

- run

        ./examples/example_video.sh

This will load all video frames, process them one after another (with temporal regularization), save the results to *./images_output/images/video_frames*, and display input and output frames side by side. Press any key to swith to the next displayed frame.

## 3 MATLAB interface

To use the **MATLAB wrapper**,
- make sure MATLAB is installed ;-)
- in the *Makefile*, replace the *MATLAB_DIR* variable with the MATLAB directory on your machine.
- compile the code. This will place the MEX-file into *./examples/matlab*
- run MATLAB and switch to the *examples/matlab* directory.
- run any of the examples to see how the wrapper is used, e.g. type

        example1

The MATLAB wrapper does not need OpenCV to be installed.


## 4 All parameters
```
    ./main   [-i <string>]  [-cam <bool>]  [-row1d <int>] 
             [-lambda <float>]  [-alpha <float>]  [-temporal <float>]
             [-weight <bool>]  [-adapt_params <bool>]  
             [-save <string>]  [-show <bool>]  [-edges <bool>]
             [-engine <cpu|cuda>]  [-use_double <bool>]
             [-iterations <int>]  [-stop_eps <float>]  [-stop_k <int>]
             [-verbose <bool>]  [-h]
```

### Parameters:

```
-i <string>
    One or more input images to process.
    - One input image: a GUI is started where you can alter the parameters
    and see its effects on the result.
    - More than one input image: All images are first processed with the given
    parameters, and then displayed one after another.
    - No input image: a default image is chosen (images/hepburn.png).

-cam <bool>
    Use live camera stream as input. (The camera must be recognizable with OpenCV,
    for instance, this might not work on OS X).
    Default: false.

-row1d <int>
    Process only a specific row of the input images,
    using the 1D version of the Mumford-Shah functional.
    The input and result will be visualized as a graph plot.
    Default: -1 (processing as 2d image).

-lambda <float>
    The parameter of the Mumford-Shah functional
    giving the weight of the discontinuity penalization.
    Default: 0.1.

-alpha <float>
    The parameter of the Mumford-Shah functional giving the
    weight of the smoothness penalization.
    Large alpha produces a more piecewise constant (cartoon-like) looking result.
    Alpha = infinity leads to the segmentation special case,
    you can set this value using "-alpha -1".
    Default: 20.

-temporal <float>
    Parameter for temporal regularization when processing video or live camera
    frames. for higher values the result for each frame will be more and more
    similar to the result of the previous frame.
    Default: 0 (no temporal regularization).

-weight <bool>
    Whether to use image edge adaptive Mumford-Shah penalization.
    Less smoothing will be applied in pixels where the absolute value of the input
    image gradient is large.
    Default: false.

-adapt_params <bool>
    Whether to adapt the parameters 'lambda' and 'alpha' to image size.
    For one and the same input image, the solution look more or less the same
    independent of its size. The specified 'lambda' and 'alpha' are regarded
    as being the actual parameters for image size 640 x 480,
    and are suitably scaled for all other sizes.
    Default: false.

-save <string>
    Directory to store the result images.
    Default: "./images_output".

-show <bool>
    Whether to display the input and result images (in addition to saving them).
    Default: true.

-edges <bool>
    Visualize the edges (discontinuities) of the result image
    (as overlaid black lines).
    Default: false.

-engine <cpu|cuda>
    Whether to use the CPU or the CUDA implementation.
    Use "-engine cpu" or "-engine 0" for the CPU version,
    and "-engine cuda" or "-engine 1" for the CUDA version. 
    Default: CUDA.

-use_double <bool>
    Whether to use to double precision for all computations
    ('double' instead of 'float').
    Double precision is slower (especially on the GPU), but should be turned on for
        - very small '-stop_eps' values (< 10^-6),
        - a huge number of iterations (> 100000),
        - or when computing an accurate energy value.
    Default: false.

-iterations <int>
    The maximal number of primal-dual iterations.
    This is only an upper bound on the actual number of performed iterations,
    since the iterations are stopped when the solutions do not
    change significantly anymore (see "-stop_eps" and "-stop_k")
    Default: 10000.

-stop_eps <float>
    Determines the stopping criterion:
    The iterations will be stopped once the average per-pixel value change
    of the solution w.r.t. the previous iterations is smaller than this parameter.
    Default: 5.5e-5.

-stop_k <int>
    For efficiency, the stopping criterion is checked only every k-th iterations.
    Default: 10.

-verbose <bool>
    Print various information such as parameters, run time, and energy.
    Default: true.

-h, -help,
    Display usage information and exit.
```

> #### "bool" parameters are set to:
- *true* for values "1", "true", or "yes", or if the parameter is specified without a value (e.g. only "-weight")
- *false* for values "0", "false", or "no".



5 License
===========

fastms is licensed under the GNU General Public License Version 3 (GPLv3), see http://www.gnu.org/licenses/gpl.html, with an additional request:

If you make use of the library in any form in a scientific publication, please refer to https://github.com/tum-vision/fastms.git and cite the paper

```
@inproceedings{Strekalovskiy-Cremers-eccv14,
  author = {E. Strekalovskiy and D. Cremers},
  title = {Real-Time Minimization of the Piecewise Smooth Mumford-Shah Functional},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2014},
  pages = {127-141},
}
```


