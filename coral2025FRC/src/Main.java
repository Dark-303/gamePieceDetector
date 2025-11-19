import org.opencv.core.Core;
import ai.onnxruntime.OrtEnvironment;

public class Main {
    public static void main(String[] args) throws Exception {
        // Load native DLLs
        System.load(new java.io.File("natives/opencv_java4120.dll").getAbsolutePath());
        System.load(new java.io.File("natives/onnxruntime_providers_shared.dll").getAbsolutePath());

        // Test OpenCV
        System.out.println("OpenCV version: " + Core.VERSION);

        // Test ONNX Runtime
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        System.out.println("ONNX Runtime loaded successfully!");
    }
}
