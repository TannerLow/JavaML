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
    public Matrix getPreActivationValues() {
        return preActivation;
    }

    @Override
    public void connect(Layer previousLayer) {
        weights = new Matrix(previousLayer.getSize(), size);
        biases = new Matrix(1, size);
    }

    @Override
    public void connect(int inputSize) {
        weights = new Matrix(inputSize, size);
        biases = new Matrix(1, size);
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

    @Override
    public void setWeights(Matrix newWeights) throws DimensionsMismatchException {
        if(weights.rows != newWeights.rows || weights.cols != newWeights.cols) {
            int[] dimensionsA = {weights.rows, weights.cols};
            int[] dimensionsB = {newWeights.rows, newWeights.cols};
            throw new DimensionsMismatchException(dimensionsA, dimensionsB);
        }

        weights = newWeights;
    }

    @Override
    public void setBiases(Matrix newBiases) throws DimensionsMismatchException {
        if(biases.rows != newBiases.rows || biases.cols != newBiases.cols) {
            int[] dimensionsA = {biases.rows, biases.cols};
            int[] dimensionsB = {newBiases.rows, newBiases.cols};
            throw new DimensionsMismatchException(dimensionsA, dimensionsB);
        }

        biases = newBiases;
    }
}
