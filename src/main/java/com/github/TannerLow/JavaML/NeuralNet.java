package com.github.TannerLow.JavaML;

import com.github.TannerLow.JavaMatrixMath.Exceptions.DimensionsMismatchException;
import com.github.TannerLow.JavaMatrixMath.Matrix;

import java.util.LinkedList;
import java.util.List;

public class NeuralNet {

    public final int inputSize;
    public final List<Layer> layers;

    public NeuralNet(int inputSize) {
        this.inputSize = inputSize;
        layers = new LinkedList<>();
    }

    public void addLayer(Layer layer) {
        layers.add(layer);
    }

    public void compile() {
        if(!layers.isEmpty()) {
            layers.get(0).connect(inputSize);
        }

        for(int i = 1; i < layers.size(); i++) {
            layers.get(i).connect(layers.get(i-1));
        }

        randomizeWeightsAndBiases();
    }

    public void randomizeWeightsAndBiases() {
        for(Layer layer : layers) {
            // randomize the weights
            Matrix weights = layer.getWeights();
            Matrix newWeights = new Matrix(weights.rows, weights.cols);
            for(int i = 0; i < newWeights.data.length; i++) {
                newWeights.data[i] = (float) (Math.random() * 2 - 1);
            }
            layer.setWeights(newWeights);

            // randomize the biases
            Matrix biases = layer.getBiases();
            Matrix newBiases = new Matrix(biases.rows, biases.cols);
            for(int i = 0; i < newBiases.data.length; i++) {
                newBiases.data[i] = (float) (Math.random() * 2 - 1);
            }
            layer.setBiases(newBiases);
        }
    }

    public Matrix predict(Matrix input) throws DimensionsMismatchException {
        if(input.cols != inputSize) {
            int[] dimensionsA = {input.cols};
            int[] dimensionsB = {inputSize};
            throw new DimensionsMismatchException(dimensionsA, dimensionsB);
        }

        if(layers.size() == 0) {
            return input;
        }

        for(Layer layer : layers) {
            input = layer.process(input);
        }

        return input;
    }

    public Matrix predictVerbose(Matrix input) {
        if(layers.size() == 0) {
            return null;
        }

        for(Layer layer : layers) {
            System.out.println("\nLayer Input:");
            for(float x : input.data) {
                System.out.print(x + " ");
            }
            System.out.println("\nLayer Weights:");
            for(float x : layer.getWeights().data) {
                System.out.print(x + " ");
            }
            System.out.println("\nLayer Biases:");
            for(float x : layer.getBiases().data) {
                System.out.print(x + " ");
            }
            input = layer.process(input);
            System.out.println("\nPre-activation Values:");
            for(float x : layer.getPreActivationValues().data) {
                System.out.print(x + " ");
            }
            System.out.println("\nLayer Output:");
            for(float x : input.data) {
                System.out.print(x + " ");
            }
            System.out.println();
        }

        return input;
    }
}
