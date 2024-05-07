package com.github.TannerLow.JavaML;

import com.github.TannerLow.JavaMatrixMath.Matrix;

import java.util.LinkedList;
import java.util.List;

public class NeuralNet {

    public final List<Layer> layers;

    public NeuralNet() {
        layers = new LinkedList<>();
    }

    public void addLayer(Layer layer) {
        if(layers.isEmpty()) {
            layers.add(layer);
        }
        else {
            layers.get(layers.size() - 1).connect(layer);
        }
    }

    public Matrix predict(Matrix input) {
        if(layers.size() == 0) {
            return null;
        }

        for(Layer layer : layers) {
            input = layer.process(input);
        }

        return input;
    }
}
