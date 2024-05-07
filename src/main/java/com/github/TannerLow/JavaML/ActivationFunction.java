package com.github.TannerLow.JavaML;

import com.github.TannerLow.JavaMatrixMath.GPU;
import com.github.TannerLow.JavaMatrixMath.Matrix;

public interface ActivationFunction {
    Matrix calculate(Matrix input);
    Matrix calculate(GPU gpu, Matrix input);
}
