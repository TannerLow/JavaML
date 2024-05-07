package com.github.TannerLow.JavaML;

import com.github.TannerLow.JavaMatrixMath.GPU;
import com.github.TannerLow.JavaMatrixMath.Matrix;

public class Relu implements ActivationFunction {

    public Relu() {

    }

    @Override
    public Matrix calculate(Matrix input) {
        return input.relu();
    }

    @Override
    public Matrix calculate(GPU gpu, Matrix input) {
        return input.relu(gpu);
    }
}
