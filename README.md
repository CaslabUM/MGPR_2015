# MGPR_2015

The code in this repository is intended to replicate Figure 4 in the paper "Efficient Multiscale Gaussian Process Regression Using Hierarchial Clustering." Questions and concerns about the code should be directed towards Ze Jia Zhang (zzejia at umich dot edu).

### Files
* Figure4.m: script to produce the desired figure.
* Train_Kern_Std.m: training algorithm for standard GPR.
* Test_Kern_Std.m: testing algorithm for standard GPR.
* Train_fd_Multiscale_F1i.m: training algorithm for MGPR.
* Test_fd_Multiscale_F1c.m: testing algorithm for MGPR.
* optLML_Multiscale2.m: optimization container for hyperparameters.
* hcluster0.m: clustering algorithm for MGPR.
* GaussMx.m, GaussMxnd.m: construction of Gaussian kernels.
* Dist2.m: compute pairwise distance between points.

