This note shows the benchmark of different CNN architecture on mobile devices. Our setting is the same as the [GEMM android tutorial] (https://github.com/strin/gemm-android) and [GEMM android benchmark] (https://github.com/strin/mocha-gemm-profile).

To simplify the benchmark, 

* our code only takes `deploy.prototxt` of the model.
* it does not read weights.

The code for the benchmark is in our fork of [Caffe OpenCL] (https://github.com/strin/caffe-opencl). To steer through experimentation, our implementation focus on **iteration speed**, not **code quality**. Once we have a deep understanding of CNN energy-performance tradeoffs, we'd like to re-write the library.

# How to use it?

```
./caffe-profile --mode cpu --model deploy.prototxt --iterations 100
```

| Parameter        | Description      | Examples |
| ------------- |:-------------:|:-----:|
| mode | the device to run the model | gpu, cpu, viennacl |
| model | the prototxt file specifying model architecture | deploy.txt |
| iterations | the number of iterations to repeat the experiment | 100 |

**note on iterations**

To get a stable metric, we run $k$ iterations such that the results of `--iterations k` and `--iterations 2k` are within relative error of 5%.

**note on device**

* GPU: our custom implementation of opencl kernels highly optimized for Samsung Galaxy S6. The goal is to reach maximum GFlops. See [GEMM android tutorial] (https://github.com/strin/gemm-android).
* CPU: multi-threaded implementation with `turbo boost` off.
* ViennaCL: use ViennaCL as OpenCL BLAS library.



### AlexNet

`bvlc_reference_net` taken from caffe: [refnet.prototxt](refnet.prototxt).

| Device        | Result      |
| ------------- |:-------------:|
| GPU | time = 0.132884 secs <br> energy = 0.01188 mJ <br> power = 0.0894015 m | 
| CPU | time = 0.19119 secs <br> energy = 0.1095 mJ <br> power = 0.572729 mW |
| ViennaCL | time = 0.631824 secs <br> energy = 0.01738 mJ <br> power = 0.0275076 mW 


**filters and GEMM**

```
conv 3x227x227 11x11 3x55x55
gemm  96x363x3025

conv 96x27x27 5x5 96x27x27
gemm  128x1200x729
gemm  128x1200x729

conv 256x13x13 3x3 256x13x13
gemm  384x2304x169

conv 384x13x13 3x3 384x13x13
gemm  192x1728x169
gemm  192x1728x169

conv 384x13x13 3x3 384x13x13
gemm  128x1728x169
gemm  128x1728x169

fc-gemm  1x9216x4096
fc-gemm  1x4096x4096
fc-gemm  1x4096x1000
```


**Layerwise Profile CPU**

```
forward conv1 : 12.5883 ms
forward relu1 : 0.933959 ms
forward pool1 : 6.58004 ms
forward norm1 : 1.91187 ms
forward conv2 : 25.5265 ms
forward relu2 : 0.629709 ms
forward pool2 : 3.28458 ms
forward norm2 : 1.21692 ms
forward conv3 : 15.2474 ms
forward relu3 : 0.204708 ms
forward conv4 : 12.5208 ms
forward relu4 : 0.208917 ms
forward conv5 : 13.204 ms
forward relu5 : 0.139625 ms
forward pool5 : 0.7615 ms
forward fc6 : 39.6844 ms
forward relu6 : 0.018666 ms
forward drop6 : 0.002583 ms
forward fc7 : 28.1235 ms
forward relu7 : 0.021667 ms
forward drop7 : 0.003417 ms
forward fc8 : 6.67329 ms
forward prob : 0.067375 ms
```

**Layerwise Profile GPU**

```
forward conv1 : 26.9266 ms
forward relu1 : 1.40379 ms
forward pool1 : 0.149667 ms
forward norm1 : 0.117583 ms
forward conv2 : 24.7577 ms
forward relu2 : 1.13658 ms
forward pool2 : 0.133083 ms
forward norm2 : 0.063708 ms
forward conv3 : 29.1167 ms
forward relu3 : 0.79675 ms
forward conv4 : 11.2297 ms
forward relu4 : 0.835291 ms
forward conv5 : 8.255 ms
forward relu5 : 0.427166 ms
forward pool5 : 0.10925 ms
forward fc6 : 33.4265 ms
forward relu6 : 1.42875 ms
forward drop6 : 0.015125 ms
forward fc7 : 16.3903 ms
forward relu7 : 0.60725 ms
forward drop7 : 0.014583 ms
forward fc8 : 5.10496 ms
forward prob : 0.178542 ms
```

**Layerwise Profile ViennaCL**

```
forward conv1 : 137.589 ms
forward relu1 : 1.22554 ms
forward pool1 : 0.129792 ms
forward norm1 : 0.07925 ms
forward conv2 : 187.883 ms
forward relu2 : 1.27942 ms
forward pool2 : 0.132333 ms
forward norm2 : 0.083167 ms
forward conv3 : 146.643 ms
forward relu3 : 1.04054 ms
forward conv4 : 111.252 ms
forward relu4 : 1.02508 ms
forward conv5 : 88.5341 ms
forward relu5 : 0.994958 ms
forward pool5 : 0.136542 ms
forward fc6 : 33.9114 ms
forward relu6 : 0.850875 ms
forward drop6 : 0.016833 ms
forward fc7 : 16.9238 ms
forward relu7 : 0.938166 ms
forward drop7 : 0.0165 ms
forward fc8 : 5.42821 ms
forward prob : 0.188333 ms
```

**ReLU Sparsity**

For every ReLU layer, we can print `#non-zero / #total count`.

```
relu 30250 / 290400 = 10.4%
relu 186624 / 186624 = 100.0%
relu 6836 / 64896 = 10.5%
relu 63923 / 64896 = 98.5%
relu 520 / 43264 = 1.4%
relu 4096 / 4096 = 100.0
relu 1 / 4096 = 0.02%
```

* If we can exploit this sparsity, the AlexNet forward pass on GPU would be reduced to `52.3 ms`. 
* We can also enforce ReLU sparsity in model. In other words, make ReLU firing pattern depend on data.

### AlexNet-4layer

`bvlc_reference_net` taken from caffe with the final convolution layer cut off: [refnet.prototxt](refnet-4layer.prototxt).

| Device        | Result      |
| ------------- |:-------------:|
| GPU | 	time = 0.126203 secs <br> energy = 0.15893 mJ <br> 	power = 1.25932 mW | 
| CPU | time = 0.171447 secs <br> energy = 0.09937 mJ <br> power = 0.579596 mW |
| ViennaCL | time = 0.532426 secs <br> energy = 0.01108 mJ <br> power = 0.0208104 mW |
