package com.github.TannerLow.JavaML;

import com.github.TannerLow.JavaMatrixMath.Matrix;

public interface Layer {
    int getSize();

    Matrix getWeights();

    Matrix getBiases();

    Matrix getPreActivationValues();

    void connect(Layer previousLayer);

    void connect(int inputSize);

    Matrix process(Matrix input);

    void setWeights(Matrix newWeights);

    void setBiases(Matrix newBiases);

    Layer copy();
}
