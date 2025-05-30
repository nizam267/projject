#include <opencv2/opencv.hpp>
#include <iostream>
#include <pthread.h>
#include <chrono>

using namespace std;
using namespace cv;

struct ThreadData {
    Mat* input;
    Mat* output;
    int start_row;
    int end_row;
    int width;
    int height;
};

// 3x3 Gaussian kernel
float kernel[3][3] = {
    {1/16.f, 2/16.f, 1/16.f},
    {2/16.f, 4/16.f, 2/16.f},
    {1/16.f, 2/16.f, 1/16.f}
};

void* apply_gaussian_blur(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    Mat& input = *(data->input);
    Mat& output = *(data->output);

    for (int y = data->start_row; y < data->end_row; ++y) {
        if (y == 0 || y == data->height - 1) continue;  // skip borders
        for (int x = 1; x < data->width - 1; ++x) {
            float sum = 0.0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    sum += kernel[ky + 1][kx + 1] * input.at<uchar>(y + ky, x + kx);
                }
            }
            output.at<uchar>(y, x) = static_cast<uchar>(sum);
        }
    }

    pthread_exit(nullptr);
}

int main() {
    Mat input = imread("input.jpg", IMREAD_GRAYSCALE);
    if (input.empty()) {
        cerr << "Error loading image!" << endl;
        return -1;
    }

    int width = input.cols;
    int height = input.rows;

    Mat output = Mat::zeros(height, width, CV_8UC1);

    int num_threads = 4;
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];

    int rows_per_thread = height / num_threads;

    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < num_threads; ++i) {
        thread_data[i] = {
            &input,
            &output,
            i * rows_per_thread,
            (i == num_threads - 1) ? height : (i + 1) * rows_per_thread,
            width,
            height
        };
        pthread_create(&threads[i], nullptr, apply_gaussian_blur, &thread_data[i]);
    }

    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << "Parallel Execution Time (" << num_threads << " threads): " << duration.count() << " seconds" << endl;

    imwrite("output_parallel.jpg", output);

    return 0;
}