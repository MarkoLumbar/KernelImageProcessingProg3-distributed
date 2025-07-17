public class Kernel {
    //vir: https://en.wikipedia.org/wiki/Kernel_(image_processing)
    //mirroring?

    public static final double[][] EDGE_DETECTION={
            {-1, -1, -1},
            {-1, 8, -1},
            {-1, -1, -1}
    };

    public static final double[][] SHARPEN={
            {0, -1, 0},
            {-1, 5, -1},
            {0, -1, 0}
    };

    public static final double[][] BLUR={
            {1.0/9, 1.0/9, 1.0/9},
            {1.0/9, 1.0/9, 1.0/9},
            {1.0/9, 1.0/9, 1.0/9}
    };

    public static final double[][] GAUSSIAN_BLUR_3={
            {1.0/16, 1.0/8, 1.0/16},
            {1.0/8, 1.0/4, 1.0/8},
            {1.0/16, 1.0/8, 1.0/16}
    };
}
