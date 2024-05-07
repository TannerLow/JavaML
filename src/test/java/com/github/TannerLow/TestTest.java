package com.github.TannerLow;

import com.github.TannerLow.JavaMatrixMath.GPU;

public class TestTest {
    public static void main(String[] args) {
        System.out.println("Testing");

        GPU gpu = new GPU();
        try (gpu){
            gpu.initialize(true);
        }
        System.out.println(gpu.isInitialized());
    }
}
