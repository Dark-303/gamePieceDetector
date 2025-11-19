import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.util.*;

public class Detector {
    private OrtEnvironment env;
    private OrtSession session;

    private int inputWidth = 640;
    private int inputHeight = 640;

    public Detector(String modelPath) throws OrtException {
        // Load ONNX Runtime environment and model
        env = OrtEnvironment.getEnvironment();
        session = env.createSession(modelPath);
        System.out.println("ONNX model loaded!");
    }

    public List<Rect> detect(Mat frame) throws OrtException {
        // Resize frame to YOLOv5 input size
        Mat resized = new Mat();
        Imgproc.resize(frame, resized, new Size(inputWidth, inputHeight));
        resized.convertTo(resized, CvType.CV_32F, 1.0 / 255); // normalize 0-1

        // Convert Mat to FloatBuffer in CxHxW order
        float[] data = new float[3 * inputWidth * inputHeight];
        int index = 0;
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < inputHeight; i++) {
                for (int j = 0; j < inputWidth; j++) {
                    double[] pixel = resized.get(i, j);
                    data[index++]
