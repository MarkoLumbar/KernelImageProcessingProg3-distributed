import java.awt.*;
import java.awt.image.BufferedImage;

public class ImgProcessor {

    public static BufferedImage convolution (BufferedImage image, double[][] kernel){
        System.out.println("Convolution");
        int widthColumns = image.getWidth();
        int heightRows = image.getHeight();
        int imageType = image.getType();
        int kernelWidthColumns = kernel.length;
        int kernelHeightRows = kernel[0].length;
        int kernelOriginX = kernelWidthColumns / 2; //center kernela po x osi
        int kernelOriginY = kernelHeightRows / 2;

        if (imageType == 0) { //falback za unkown image type
            imageType = BufferedImage.TYPE_INT_RGB;
        }

        BufferedImage newImage = new BufferedImage(widthColumns, heightRows, imageType);

        //naivna implementacija iz wikija: https://en.wikipedia.org/wiki/Kernel_(image_processing)
        //loopamo čez sliko in ignoriramo rob, slika bo zato malenkost manjsa
        for (int y = kernelOriginY; y < heightRows - kernelOriginY; y++) {
            for (int x = kernelOriginX; x <widthColumns - kernelOriginX ; x++) {

                double redAcc = 0, greenAcc = 0, blueAcc = 0;

                //loopamo kernel
                for (int ky = 0; ky < kernelHeightRows; ky++) {
                    for (int kx = 0; kx < kernelWidthColumns; kx++) {
                        int imageX = x + kx - kernelOriginX;
                        int imageY = y + ky - kernelOriginY;

                        //http://tech.abdulfatir.com/2014/05/kernel-image-processing.html for extracting RGB and truncating values
                        int rgb = image.getRGB(imageX, imageY);
                        int red = (rgb >> 16) & 0xFF;
                        int green = (rgb >> 8) & 0xFF;
                        int blue = (rgb & 0xFF);

                        redAcc += red * kernel[ky][kx];
                        greenAcc += green * kernel[ky][kx];
                        blueAcc += blue * kernel[ky][kx];

                    }
                }
                int truncatedRed, truncatedGreen, truncatedBlue; //ker values so lahko samo od 0-255
                truncatedRed = Math.max(0, Math.min(255, (int) redAcc));
                truncatedGreen = Math.max(0, Math.min(255, (int) greenAcc));
                truncatedBlue = Math.max(0, Math.min(255, (int) blueAcc));

                Color color = new Color(truncatedRed, truncatedGreen, truncatedBlue);
                newImage.setRGB(x, y, color.getRGB()); //nastavimo barvo pixla v novi sliki
            }

        }

        return newImage;
    }
}
