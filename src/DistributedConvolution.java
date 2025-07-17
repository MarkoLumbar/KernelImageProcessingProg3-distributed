import mpi.MPI;



public class DistributedConvolution {
    public static void main(String[] args) {
        MPI.Init(args);

        int rank = MPI.COMM_WORLD.Rank();
        int size = MPI.COMM_WORLD.Size();

        String input_path = null;
        String output_path = null;
        String kernel_name = null;

        if (rank == 0){
            if (args.length < 3) { //premalo argumentov
                System.err.println("Usage: mpjrun.sh -np <st_proc> <input_path> <output_path> <kernel_name>");
                MPI.Finalize(); //procesi naj se končajo
                return;
            }
            input_path = args[0];
            output_path = args[1];
            kernel_name = args[2];

        }

        double [][] kernel = null;
        double [] kernel_flat = new double[9]; //za Bcast

        if (rank == 0){
            switch (kernel_name.toLowerCase()){
                case "edge":
                    kernel = Kernel.EDGE_DETECTION;
                    break;
                case "sharpen":
                    kernel = Kernel.SHARPEN;
                    break;
                case "blur":
                    kernel = Kernel.BLUR;
                    break;
                case "gaussian_blur_3":
                    kernel = Kernel.GAUSSIAN_BLUR_3;
                    break;
                default:
                    System.err.println("Unkown kernel: " + kernel_name + "usiing default kernel: sharpen");
                    kernel = Kernel.SHARPEN;
            }

            //flatten the kernel za Bcast() v 1D
            int k = 0;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    kernel_flat[k++] = kernel[i][j];
                }
            }
        }

        //master broadcasta kernel vsem procesom
        MPI.COMM_WORLD.Bcast(kernel_flat, 0, 9, MPI.DOUBLE, 0);

        //vsak proces pretvori kernel nazaj v 2D
        double[][] receivedKernel = new double[3][3];
        int k = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                receivedKernel[i][j] = kernel_flat[k++];
            }
        }


        MPI.Finalize(); //končaj MPI
    }

}
