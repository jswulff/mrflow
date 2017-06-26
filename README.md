MR-FLOW: Optical Flow for Mostly Rigid Scenes
=============================================

This software package contains the MR-Flow optical flow algorithm, as described in Wulff, Sevilla, and Black, "Optical Flow for Mostly Rigid Scenes" (CVPR 2017).

MR-Flow splits the scene into moving objects and the static scene, uses general optical flow for the moving parts, and strong geometric constraints in the static scene.

Note that due to some small bugfixes and changes in the codebase, the results can differ slightly from those given in the paper.

We hope this software is useful to you.
If you have any questions, comments, or issues, please do not hesitate to [contact us](mailto:jonas.wulff@tuebingen.mpg.de).


Installation
------------

### Requirements

* [OpenCV](http://www.opencv.org) >= 3.0 with Python bindings, including the [contrib packages](https://github.com/opencv/opencv_contrib).
* Numpy
* Scipy
* Matplotlib
* Cython
* Chumpy

Except OpenCV, all other packages can be installed via PIP.



### Installation

To use MR-Flow, please set the $MRFLOW_HOME environment variable to the root directory (where this file is located).

MR-Flow uses [Karteek Alahari's Efficient TRWS code](https://lear.inrialpes.fr/people/alahari/code/eff_mlabel.html), which is included in this package.
To compile the Cython extension, simply run `build_trws.sh` in the main directory.

After that, you can test MR-Flow as

    python mrflow.py --no_init example/frame1.png example/frame2.png example/frame3.png



Usage
--------

The recommended way to call MR-Flow as

    python mrflow.py \
      --rigidity RIGIDITYMAP.png \
      --flow_fwd FLOW_FWD.flo \
      --flow_bwd FLOW_BWD.flo \
      --backflow_fwd BACKFLOW_FWD.flo \
      --backflow_bwd BACKFLOW_BWD.flo \
      IMAGE1 IMAGE2 IMAGE3
    
This will compute the optical flow from IMAGE2 to IMAGE3.
MR-Flow will save the resulting optical flow, the structure, and some visualizations to the current directory.

Some important flags are

* `--rigidity RIGIDITYMAP` : Use pre-computed rigidity, for example given by a semantic segmentation CNN. RIGIDITYMAP has to be a monochrome PNG, where black denotes a moving object and white denotes the static scene. Note that the rigiditymap does not have to be binary; values between 0 and 1 are interpreted as probabilties for a pixel to be part of the static scene.
* `--flow_fwd FLOW_FWD` : Initial flow from IMAGE2 to IMAGE3. Note that if not all initial flow fields are given, the initial flow is computed using DiscreteFlow.
* `--flow_bwd FLOW_BWD` : Initial flow from IMAGE2 to IMAGE1.
* `--backflow_fwd BFLOW_FWD` : Initial flow from IMAGE3 to IMAGE2.
* `--backflow_bwd BFLOW_BWD` : Initial flow from IMAGE1 to IMAGE2.
* `--no_init`  : Omit the initializations that are not given. Instead, use DiscreteFlow to compute the initial optical flow maps, and use a uniform prior for the rigidity estimate. If both `--no_init` is given and some initialization is provided (e.g. by calling `python mrflow.py --no_init --rigidity RIGIDITMAP.png ...`), the initialization that is given is still used. **IF YOU OMIT THE INITIALIZATION, DO NOT EXPECT COMPARABLE PERFORMANCE TO THE PAPER.**
* `--tempdir TDIR` : Alternative output directory 
* `--override_optimization 1` : Do not perform variational structure refinement. Often this gives sufficient accuracy, at a much lower computational cost.


If you want to run MR-Flow without initialization, 



For further flags and parameters, please see the file `parameter.py`.


Citation
--------

If you use MR-Flow, please cite the following paper: 

    @inproceedings{Wulff:CVPR:2017,
    title = {Optical Flow in Mostly Rigid Scenes},
    author = {Wulff, Jonas and Sevilla-Lara, Laura and Black, Michael J.},
    booktitle = {IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
    month = jul,
    year = {2017},
    month_numeric = {7}
    }


License
-------

See `LICENSE.md` for licensing issues and details.

This packages contains code from the paper
    
    "Reduce, Reuse & Recycle: Efficiently Solving Multi-Label MRFs"
    Karteek Alahari, Pushmeet Kohli, Philip H. S. Torr
    In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2008. 
    
The code is contained in the directory `extern/eff_trws`.
For licensing details regarding this code, please see `extern/eff_trws/eff_mlabel_ver1/README.txt`.


Contact
-------

If you run into any issues with MR-Flow, please do not hesitate to contact us at
`jonas.wulff@tuebingen.mpg.de`.
