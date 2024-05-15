package com.github.TannerLow;

import com.github.TannerLow.JavaML.DenseLayer;
import com.github.TannerLow.JavaML.Layer;
import com.github.TannerLow.JavaML.NeuralNet;
import com.github.TannerLow.JavaML.Relu;
import com.github.TannerLow.JavaML.Softmax;
import com.github.TannerLow.JavaMatrixMath.GPU;
import com.github.TannerLow.JavaMatrixMath.InternalFile;
import com.github.TannerLow.JavaMatrixMath.Matrix;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

public class TestTest {
    public static void main(String[] args) throws IOException {
        System.out.println("Testing");

        GPU gpu = new GPU();
        try (gpu){
            // Load GPU program code into memory
            String matricesKernelFilePath = "kernels/Matrices.cl";
            String matricesKernelCode = readFromInternalFile(matricesKernelFilePath);
            if(matricesKernelCode == null) {
                throw new IOException("Failed to read file: " + matricesKernelFilePath);
            }

            gpu.initialize(true);
            int programId = gpu.loadProgram(matricesKernelCode);
            gpu.loadKernel(programId, "Matrices", "matrixMultiply");
            gpu.loadKernel(programId, "Matrices", "addRowToRows");
            gpu.loadKernel(programId, "Matrices", "relu");
            gpu.loadKernel(programId, "Matrices", "softmax");

            NeuralNet net = new NeuralNet(2);
            Layer inputLayer = new DenseLayer(2, new Relu(), gpu);
            net.addLayer(inputLayer);
            Layer hiddenLayer = new DenseLayer(2, new Softmax(), gpu);
            net.addLayer(hiddenLayer);
            net.compile();

            float[] inputData = {1, 0};
            Matrix input = new Matrix(1, 2, inputData);
            Matrix output = net.predictVerbose(input);

            System.out.println("\nModel Output:");
            for(int i = 0; i < output.data.length; i++) {
                System.out.println(output.data[i]);
            }
        }
    }

    private static String readFromInternalFile(String filepath) {
        try(InputStream fileInputStream = InternalFile.getInstance().getFileInputStream(filepath)) {
            byte[] bytes = fileInputStream.readAllBytes();
            String fileContent = new String(bytes, StandardCharsets.UTF_8);
            return fileContent;
        } catch (Exception e) {
            e.printStackTrace();
        }

        return null;
    }
}
