# IntegralCalc
Definite integration on GPU (up to triple integrals). Made as a GPU programming project for University.
## Compiling
For Linux, install the OpenCL SDK:

**sudo apt-get install ocl-icd-opencl-dev**

And an OpenCL platform:

- **sudo apt-get install beignet-opencl-icd** for Intel iGPUs
- **sudo apt-get install nvidia-opencl-icd** for NVIDIA GPUs
- **sudo apt-get install mesa-opencl-icd** for AMD GPUs
- **sudo apt-get install pocl-opencl-icd** for the CPU

For Mac OS X you *should* be good to go.

Finally just run **sh build.sh** on Linux and **sh buildMac.sh** on Mac OS X.

## Plotting
You can see the plots in pam format with imageMagick:

**sudo apt-get install imagemagick**

## Integration methods
The methods used were:
- Monte Carlo Sample-Mean (with antithetic variates)
- Monte Carlo Hit or Miss (with antithetic variates)
- Rectangle rule
- Trapezoid rule
- Simpson Rule
- Rectangle rule with random rectangles (and antithetic variates)

## License
IntegralCalc is released as GPLv3

## Credits
- David B. Tomas for the MWC64X GPU RNG
- Josura for GPU sorting
