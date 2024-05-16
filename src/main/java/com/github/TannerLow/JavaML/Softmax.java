package com.github.TannerLow.JavaML;

import com.github.TannerLow.JavaMatrixMath.GPU;
import com.github.TannerLow.JavaMatrixMath.Matrix;

public class Softmax implements ActivationFunction {

    public Softmax() {

    }

    @Override
    public Matrix calculate(Matrix input) {
        return input.verticalSoftmax();
    }

    @Override
    public Matrix calculate(GPU gpu, Matrix input) {
        return input.verticalSoftmax(gpu);
    }
}
