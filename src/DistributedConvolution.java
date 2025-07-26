import mpi.MPI;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;


public class DistributedConvolution {
    public static void main(String[] args) {
        MPI.Init(args);

        int rank = MPI.COMM_WORLD.Rank();
        int size = MPI.COMM_WORLD.Size(); //st procesov

        String input_path = null;
        String output_path = null;
        String kernel_name = null;

        BufferedImage input_image = null;
        int imageWidth = 0;
        int imageHeight = 0;
        int[] imageRGB = null;

        long start = 0;

        MPI.COMM_WORLD.Barrier(); //sinhronizacija procesov, istocasni zacetek

        if (rank == 0){
            start = System.nanoTime();
            if (args.length < 6) { //premalo argumentov
                System.err.println("Usage: mpjrun.sh -np <st_proc> <input_path> <output_path> <kernel_name>");
                MPI.Finalize(); //procesi naj se končajo
                return;
            }
            //prvi je rank procesa, drugi št procesov, tretji ime java fila
            input_path = args[3];
            output_path = args[4];
            kernel_name = args[5];

            try {
                input_image = ImageIO.read(new File(input_path));
                if (input_image == null) {
                    throw new IOException("Slika ni dosegljiva.");
                }
                imageWidth = input_image.getWidth();
                imageHeight = input_image.getHeight();

                imageRGB = new int[imageWidth * imageHeight]; //1D array ki hrani pixle slike 1 int = 1rgb pixel
                input_image.getRGB(0, 0, imageWidth, imageHeight, imageRGB, 0, imageWidth); //vrsticno preberemo sliko
            }catch (IOException e){
                System.err.println("Error reading image: " + e.getMessage());
                MPI.Finalize();
                return;
            }
        }


        // vsi procesi poznajo velikost slike
        int[] dimensions = new int[2];
        if (rank == 0) {
            dimensions[0] = imageWidth;
            dimensions[1] = imageHeight;
        }
        MPI.COMM_WORLD.Bcast(dimensions, 0, 2, MPI.INT, 0);
        imageWidth = dimensions[0];
        imageHeight = dimensions[1];

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

        int rowsPerProcess = imageHeight / size; //št vrstic na proces
        int remainingRows = imageHeight % size; //preostale vrstice
        int localHeight = rowsPerProcess + (rank < remainingRows ? 1 : 0); //če je rank manjši od preostalih vrstic, dobi še eno vrstico
        int [] localRGB = new int[localHeight * imageWidth]; //local buffer, kjer Scatter shrani pixle posamznega procesa

        int[] chunkSizes = null;
        int[] startPos = null;
        if (rank == 0) {
            chunkSizes = new int[size];
            startPos = new int[size];
            int offset = 0;
            for (int i = 0; i < size; i++) {
                int rows = rowsPerProcess + (i < remainingRows ? 1 : 0);
                chunkSizes[i] = rows * imageWidth;
                startPos[i] = offset;
                offset += chunkSizes[i];
            }
        }

        // Razpošiljanje podatkov iz imageRGB vsem procesom
        MPI.COMM_WORLD.Scatterv(
                imageRGB,      // vir samo rank 0 ima celotno 1D sliko
                0,
                chunkSizes, //st elm vsakemu procesu
                startPos,
                MPI.INT,
                localRGB,      // cilj - vsak proces svoj pas slike
                0,
                localRGB.length, //st elm ki jih proces pricakuje
                MPI.INT,
                0
        );

        System.out.println("Proces " + rank + " dobi " + localRGB.length + " pixlov.");


        //pretvorimo 1D tabelo v v local Bufferedimage
        BufferedImage localImage = new BufferedImage(imageWidth, localHeight, BufferedImage.TYPE_INT_RGB);
        localImage.setRGB(0, 0, imageWidth, localHeight, localRGB, 0, imageWidth);

        //konvolucija
        BufferedImage localConvolution = ImgProcessor.convolution(localImage, receivedKernel);

        //spet nazaj v 1D zaradi Gatherv
        int[] processedLocalRGB = new int[localRGB.length];
        localConvolution.getRGB(0, 0, imageWidth, localHeight, processedLocalRGB, 0, imageWidth);


        int[] finalRGB = null;
        if (rank == 0) {
            finalRGB = new int[imageWidth * imageHeight]; //buffer za koncno sliko
        }

        //zdruzimo vse pasove v finalRGB
        MPI.COMM_WORLD.Gatherv(
                processedLocalRGB,        // lokalni vir - proces to pošlje
                0,
                processedLocalRGB.length,
                MPI.INT,
                finalRGB,              // končni rezultat rank 0
                0,                      // zacetek v finalRGB bufferju
                chunkSizes,             // st elm za vsak proces
                startPos,
                MPI.INT,
                0                   //root prejme vse
        );

        MPI.COMM_WORLD.Barrier(); //sinhronizacija pred koncem

        if (rank == 0) {
            long end = System.nanoTime();
            BufferedImage outputImage = new BufferedImage(imageWidth, imageHeight, BufferedImage.TYPE_INT_RGB);
            outputImage.setRGB(0, 0, imageWidth, imageHeight, finalRGB, 0, imageWidth);
            try {
                System.out.println("Execution time (ms): " + (end - start) / 1_000_000); //da merimo samo konvolucijo
                File outputFile = new File(output_path);
                String formatName = output_path.substring(output_path.lastIndexOf('.') + 1); //najde piko in bere vse za njo - torej format
                ImageIO.write(outputImage, formatName, outputFile);
                System.out.println("Slika uspešno shranjena na: " + output_path);
            } catch (IOException e) {
                System.err.println("Napaka pri shranjevanju slike: " + e.getMessage());
            }
        }
            MPI.Finalize(); //končaj MPI
    }

}
