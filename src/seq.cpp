#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace std;
using namespace cv;

int main() {
    // Load image in grayscale
    Mat input = imread("input.jpg", IMREAD_GRAYSCALE);
    if (input.empty()) {
        cerr << "Error loading image!" << endl;
        return -1;
    }

    int width = input.cols;
    int height = input.rows;

    Mat output = Mat::zeros(height, width, CV_8UC1);

    // Define 3x3 Gaussian kernel
    float kernel[3][3] = {
        {1/16.f, 2/16.f, 1/16.f},
        {2/16.f, 4/16.f, 2/16.f},
        {1/16.f, 2/16.f, 1/16.f}
    };

    // Start time
    auto start = chrono::high_resolution_clock::now();

    // Apply Gaussian blur manually
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            float sum = 0.0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int pixel = static_cast<int>(input.at<uchar>(y + ky, x + kx));
                    sum += kernel[ky + 1][kx + 1] * pixel;
                }
            }
            output.at<uchar>(y, x) = static_cast<uchar>(sum);
        }
    }

    // End time
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << "Sequential Execution Time: " << duration.count() << " seconds" << endl;

    // Save output image
    imwrite("output_sequential.jpg", output);

    return 0;
}