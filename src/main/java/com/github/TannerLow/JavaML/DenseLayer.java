package com.github.TannerLow.JavaML;

import com.github.TannerLow.JavaMatrixMath.Exceptions.DimensionsMismatchException;
import com.github.TannerLow.JavaMatrixMath.GPU;
import com.github.TannerLow.JavaMatrixMath.Matrix;

public class DenseLayer implements Layer {

    public final int size;
    public final ActivationFunction activationFunction;
    public final GPU gpu;
    private Matrix weights;
    private Matrix biases;
    private Matrix preActivation;
    private Matrix postActivation;

    public DenseLayer(int size, ActivationFunction activationFunction) {
        this.size = size;
        this.activationFunction = activationFunction;
        this.gpu = null;
    }

    public DenseLayer(int size, ActivationFunction activationFunction, GPU gpu) {
        this.size = size;
        this.activationFunction = activationFunction;
        this.gpu = gpu;
    }

    @Override
    public int getSize() {
        return size;
    }

    @Override
    public Matrix getWeights() {
        return weights;
    }

    @Override
    public Matrix getBiases() {
        return biases;
    }

    @Override
    public void connect(Layer nextLayer) {
        weights = new Matrix(size, nextLayer.getSize());
        biases = new Matrix(1, nextLayer.getSize());
    }

    @Override
    public Matrix process(Matrix input) throws NullPointerException, DimensionsMismatchException{
        if(input == null) {
            throw new NullPointerException();
        }

        if(weights == null) {
            throw new NullPointerException();
        }

        if(input.cols != weights.rows) {
            int[] dimensionsA = {input.rows, input.cols};
            int[] dimensionsB = {weights.rows, weights.cols};
            throw new DimensionsMismatchException(dimensionsA, dimensionsB);
        }

        if(gpu != null && !Matrix.isCompatibleWithGPU(gpu)) {
            throw new IllegalStateException("GPU supplied to layer but not compatible/initialized.");
        }

        if(gpu == null) {
            preActivation = input.multiply(weights).addRowToRows(biases);
            postActivation = activationFunction.calculate(preActivation);
        }
        else {
            preActivation = input.multiply(gpu, weights).addRowToRows(gpu, biases);
            postActivation = activationFunction.calculate(gpu, preActivation);
        }

        return postActivation;
    }
}
