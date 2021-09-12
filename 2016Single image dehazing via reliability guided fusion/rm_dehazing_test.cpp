#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <algorithm>
#include <opencv2\opencv.hpp>

#include "rm_dehazing.cpp"

using namespace cv;
using namespace std;

typedef struct Pixel 
{
    int x, y;
    int data;
}Pixel;

bool structCmp(const Pixel &a, const Pixel &b) 
{
    return a.data > b.data;//descending降序
}

Mat minFilter(Mat srcImage, int kernelSize);
Mat maxFilter(Mat srcImage, int kernelSize);
void makeDepth32f(Mat& source, Mat& output);
void guidedFilter(Mat& source, Mat& guided_image, Mat& output, int radius, float epsilon);
Mat getTransmission(Mat& srcimg, Mat& transmission, int windowsize);
Mat recover(Mat& srcimg, Mat& t, float *array, int windowsize);

int main() 
{
    string loc = "F:/JZYY/pic/dehazing/5w.png" ;
    double scale = 1.0;
    string name = "forest";
    clock_t start, finish;
    double duration;

    cout << "A defog program" << endl
        << "----------------" << endl;

    Mat image = imread(loc);
    Mat resizedImage;
    int originRows = image.rows;
    int originCols = image.cols;

    imshow("hazyimg", image);

    if (scale < 1.0) 
    {
        resize(image, resizedImage, Size(originCols * scale, originRows * scale));
    }
    else 
    {
        scale = 1.0;
        resizedImage = image;
    }

    int rows = resizedImage.rows;
    int cols = resizedImage.cols;
    Mat convertImage;
    resizedImage.convertTo(convertImage, CV_32FC3, 1 / 255.0);
    int kernelSize = 15 ? max((rows * 0.01), (cols * 0.01)) : 15 < max((rows * 0.01), (cols * 0.01));
    //int kernelSize = 15;
    int parse = kernelSize / 2;
    Mat darkChannel(rows, cols, CV_8UC1);
    Mat normalDark(rows, cols, CV_32FC1);
    Mat normal(rows, cols, CV_32FC1);

    int nr = rows;
    int nl = cols;
    float b, g, r;

    start = clock();
    cout << "generating dark channel image." << endl;
    if (resizedImage.isContinuous()) 
    {
        nl = nr * nl;
        nr = 1;
    }
    for (int i = 0; i < nr; i++) 
    {
        float min;
        const uchar* inData = resizedImage.ptr<uchar>(i);
        uchar* outData = darkChannel.ptr<uchar>(i);
        for (int j = 0; j < nl; j++) 
        {
            b = *inData++;
            g = *inData++;
            r = *inData++;
            min = b > g ? g : b;
            min = min > r ? r : min;
            *outData++ = min;
        }
    }
    darkChannel = minFilter(darkChannel, kernelSize);
    darkChannel.convertTo(normal, CV_32FC1, 1 / 255.0);

    //imshow("darkChannel", darkChannel);
    cout << "dark channel generated." << endl;

    //estimate Airlight
    //开一个结构体数组存暗通道，再sort，取最大0.1%，利用结构体内存储的原始坐标在原图中取点
    cout << "estimating airlight." << endl;
    rows = darkChannel.rows, cols = darkChannel.cols;
    int pixelTot = rows * cols * 0.001;
    int *A = new int[3];
    Pixel *toppixels, *allpixels;
    toppixels = new Pixel[pixelTot];
    allpixels = new Pixel[rows * cols];

    for (unsigned int r = 0; r < rows; r++) 
    {
        const uchar *data = darkChannel.ptr<uchar>(r);
        for (unsigned int c = 0; c < cols; c++) 
        {
            allpixels[r*cols + c].data = *data;
            allpixels[r*cols + c].x = r;
            allpixels[r*cols + c].y = c;
        }
    }
    std::sort(allpixels, allpixels + rows * cols, structCmp);

    memcpy(toppixels, allpixels, pixelTot * sizeof(Pixel));

    float A_r, A_g, A_b, avg, maximum = 0;
    int idx, idy, max_x, max_y;
    for (int i = 0; i < pixelTot; i++) 
    {
        idx = allpixels[i].x; idy = allpixels[i].y;
        const uchar *data = resizedImage.ptr<uchar>(idx);
        data += 3 * idy;
        A_b = *data++;
        A_g = *data++;
        A_r = *data++;
        //cout << A_r << " " << A_g << " " << A_b << endl;
        avg = (A_r + A_g + A_b) / 3.0;
        if (maximum < avg) 
        {
            maximum = avg;
            max_x = idx;
            max_y = idy;
        }
    }

    delete[] toppixels;
    delete[] allpixels;

    for (int i = 0; i < 3; i++) 
    {
        A[i] = resizedImage.at<Vec3b>(max_x, max_y)[i];
    }

    //暗通道归一化操作（除A）
    //(I / A)
    float tmp_A[3];
    tmp_A[0] = A[0] / 255.0;
    tmp_A[1] = A[1] / 255.0;
    tmp_A[2] = A[2] / 255.0;

    cout << "airlight estimated as: " << tmp_A[0] << ", " << tmp_A[1] << ", " << tmp_A[2] << endl;
    
    int radius = 3; int kernel = 2 * radius+1;
    Size win_size(kernel, kernel);
    
    //构造权重函数S(x)
    Mat S(rows, cols, CV_32FC1);
    float w1 = 10.0; float w2 = 0.2; float min = 1.0;
    float b_A, g_A, r_A; float pixsum;

    for (int i = 0; i < nr; i++) 
    {
        const float* inData = convertImage.ptr<float>(i);
        float* outData = normalDark.ptr<float>(i);
        float* sData = S.ptr<float>(i);
        for (int j = 0; j < nl; j++) 
        {
            b = *inData++; g = *inData++; r = *inData++;

            b_A = b / tmp_A[0];
            g_A = g / tmp_A[1];
            r_A = r / tmp_A[2];

            min = b_A > g_A ? g_A : b_A;
            min = min > r_A ? r_A : min;
            *outData++ = min;

            pixsum = (b - tmp_A[0]) * (b - tmp_A[0]) + (g - tmp_A[1]) * (g - tmp_A[1]) + (r - tmp_A[2]) * (b - tmp_A[2]);
            *sData++ = exp((-1 * w1) * pixsum);
        }
    }

    //imshow("S", S);

    //calculate the Iroi map
    Mat Ic = normalDark; Mat Icest;
    Mat Imin; Mat umin; Mat Ibeta;

    Ic = Ic.mul(Mat::ones(rows, cols, CV_32FC1) - w2 * S);
    //imshow("Ic", Ic);
    Imin = minFilter(Ic, kernel);
    //imshow("Imin", Imin);
    
    boxFilter(Imin, umin, CV_32F, win_size);
    Ibeta = maxFilter(umin, kernel);
    //imshow("Ibeta", Ibeta);

    Mat ubeta; Mat uc;
    boxFilter(Ibeta, ubeta, CV_32F, win_size);
    boxFilter(Ic, uc, CV_32F, win_size);

    float fai = 0.0001; Mat Iroi;
    Mat weight = (Mat::ones(rows, cols, CV_32FC1)) * fai;
    divide((Ic.mul(ubeta)), (uc + weight), Iroi);
    //imshow("Iroi", Iroi);

    //calculate the reliability map alpha
    Mat uepsilon; Mat alpha;
    Mat m = Ibeta - umin; Mat n = Ibeta - Iroi;
    boxFilter(m.mul(m) + n.mul(n), uepsilon, CV_32F, win_size);

    float zeta = 0.0025;
    uepsilon / (uepsilon + Mat::ones(rows, cols, CV_32FC1) * zeta);
    alpha = Mat::ones(rows, cols, CV_32FC1) - uepsilon / (uepsilon + Mat::ones(rows, cols, CV_32FC1) * zeta);
    //imshow("alpha", alpha);

    //calculate the Idark map
    Mat Ialbe; Mat ualpha; Mat ualbe; Mat Idark;
    Ialbe = alpha.mul(Ibeta);

    boxFilter(alpha, ualpha, CV_32F, win_size);
    boxFilter(Ialbe, ualbe, CV_32F, win_size);

    Idark = Iroi.mul(Mat::ones(rows, cols, CV_32FC1) - ualpha) + ualbe;
    //imshow("Idark", Idark);

    float w = 0.95;
    Mat t; t = Mat::ones(rows, cols, CV_32FC1) - w*Idark;
    int kernelSizeTrans = std::max(3, kernelSize);
    Mat trans = getTransmission(convertImage, t, kernelSizeTrans);
    //imshow("t",trans);

    Mat finalImage = recover(convertImage, trans, tmp_A, kernelSize);
    cout << "recovering finished." << endl;

    Mat resizedFinal;
    if (scale < 1.0) 
    {
        resize(finalImage, resizedFinal, Size(originCols, originRows));
        imshow("final", resizedFinal);
    }
    else 
    {
        imshow("final", finalImage);
    }

    finish = clock();
    duration = (double)(finish - start);
    cout << "defog used " << duration << "ms time;" << endl;
    waitKey(0);

    finalImage.convertTo(finalImage, CV_8UC3, 255);
    imwrite("images/5Wrefined.png", finalImage);

    destroyAllWindows();
    image.release();
    resizedImage.release();
    convertImage.release();
    darkChannel.release();
    trans.release();
    finalImage.release();
    return 0;
}
