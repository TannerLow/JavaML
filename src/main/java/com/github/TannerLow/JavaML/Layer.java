package com.github.TannerLow.JavaML;

import com.github.TannerLow.JavaMatrixMath.Matrix;

public interface Layer {
    int getSize();

    Matrix getWeights();

    Matrix getBiases();

    void connect(Layer nextLayer);

    Matrix process(Matrix input);
}
