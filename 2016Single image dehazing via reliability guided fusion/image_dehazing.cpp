#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <algorithm>
#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;

float sqr(float x);
float norm(float *array);
float avg(float *vals, int n);
float conv(float *xs, float *ys, int n);
Mat stress(Mat& input);
Mat getDehaze(Mat& scrimg, Mat& transmission, float *array);
Mat getTransmission(Mat& input, float *airlight);

int main()
{
    string loc = "F:/JZYY/pic/dehazing/4w.png";
    double scale = 1.0;

    clock_t start, finish;
    double duration;

    cout << "A defog program" << endl
        << "----------------" << endl;

    Mat image = imread(loc);
    imshow("hazyiamge", image);
    cout << "input hazy image" << endl;
    Mat resizedImage;
    int originRows = image.rows;
    int originCols = image.cols;

    if (scale < 1.0)
    {
        resize(image, resizedImage, Size(originCols * scale, originRows * scale));
    }
    else
    {

        scale = 1.0;
        resizedImage = image;
    }

    start = clock();
    int rows = resizedImage.rows;
    int cols = resizedImage.cols;
    int nr = rows; int nl = cols;
    Mat convertImage(nr, nl, CV_32FC3);
    resizedImage.convertTo(convertImage, CV_32FC3, 1 / 255.0, 0);
    int kernelSize = 15;

    float tmp_A[3];
    //0.84 0.83 0.80
    //0.91 0.80 0.86

    tmp_A[0] = 0.77;
    tmp_A[1] = 0.85;
    tmp_A[2] = 0.90;

    Mat trans = getTransmission(convertImage, tmp_A);
    cout << "tansmission estimated." << endl;
    imshow("t", trans);

    cout << "start recovering." << endl;
    Mat finalImage = getDehaze(convertImage, trans, tmp_A);
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
    const char* path;
    path = "images/4Wde2.png";
    imwrite(path, finalImage);
    destroyAllWindows();
    image.release();
    resizedImage.release();
    convertImage.release();
    trans.release();
    finalImage.release();
    return 0;
}

float sqr(float x)
{
    return x * x;
}

float norm(float *array)
{
    return sqrt(sqr(array[0]) + sqr(array[1]) + sqr(array[2]));
}

float avg(float *vals, int n)
{
    float sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += vals[i];
    }
    
    return sum / n;
}

float conv(float *xs, float *ys, int n)
{
    float ex = avg(xs, n);
    float ey = avg(ys, n);
    float sum = 0;
    for (int i = 0; i < n; i++)
    {
        
        sum += (xs[i] - ex)*(ys[i] - ey);
    }
    
    return sum / n;
}

Mat getDehaze(Mat& scrimg, Mat& transmission, float *array)
{
    int nr = transmission.rows; int nl = transmission.cols;
    Mat result = Mat::zeros(nr, nl, CV_32FC3);
    Mat one = Mat::ones(nr, nl, CV_32FC1);
    vector<Mat> channels(3);
    split(scrimg, channels);

    Mat R = channels[2];
    Mat G = channels[1];
    Mat B = channels[0];

    channels[2] = (R - (one - transmission)*array[2]) / transmission;
    channels[1] = (G - (one - transmission)*array[1]) / transmission;
    channels[0] = (B - (one - transmission)*array[0]) / transmission;

    merge(channels, result);
    return result;
}

Mat getTransmission(Mat& input, float *airlight)
{
    float normA = norm(airlight);
    //Calculate Ia
    int nr = input.rows, nl = input.cols;
    Mat Ia(nr, nl, CV_32FC1);
    for (int i = 0; i < nr; i++)
    {
        const float* inPtr = input.ptr<float>(i);
        float* outPtr = Ia.ptr<float>(i);
        for (int j = 0; j < nl; j++)
        {
            float dotresult = 0;
            for (int k = 0; k < 3; k++)
            {
                dotresult += (*inPtr++)*airlight[k];
            }
            *outPtr++ = dotresult / normA;
            
        }

    }
    imshow("Ia", Ia);

    //Calculate Ir
    Mat Ir(nr, nl, CV_32FC1);
    for (int i = 0; i < nr; i++)
    {
        Vec3f* ptr = input.ptr<Vec3f>(i);
        float* irPtr = Ir.ptr<float>(i);
        float* iaPtr = Ia.ptr<float>(i);
        for (int j = 0; j < nl; j++)
        {
            float inNorm = norm(ptr[j]);
            *irPtr = sqrt(sqr(inNorm) - sqr(*iaPtr));
            if(isnan(*irPtr) != 0){
                *irPtr = 0;
            }
            iaPtr++; irPtr++;
        }
    }
    imshow("Ir", Ir);

    //Calculate h
    Mat h(nr, nl, CV_32FC1);
    for (int i = 0; i < nr; i++)
    {
        float* iaPtr = Ia.ptr<float>(i);
        float* irPtr = Ir.ptr<float>(i);
        float* hPtr = h.ptr<float>(i);
        for (int j = 0; j < nl; j++)
        {   
            *hPtr = (normA - *iaPtr) / *irPtr;
            if(isnan(*hPtr) != 0 or isinf(*hPtr) != 0){
                *hPtr = 0;
            }
            hPtr++; iaPtr++; irPtr++;
        }
    }
    imshow("h", h);

    //Estimate the eta
    int length = nr * nl;
    float* Iapix = new float[length];
    float* Irpix = new float[length];
    float* hpix = new float[length];
    for (int i = 0; i < nr; i++)
    {
        const float *IaData = Ia.ptr<float>(i);
        const float *IrData = Ir.ptr<float>(i);
        const float *hData = h.ptr<float>(i);
        for (int j = 0; j < nl; j++)
        {
            Iapix[i*nl + j] = *IaData++;
            Irpix[i*nl + j] = *IrData++;
            hpix[i*nl + j] = *hData++;
        }
    }
    
    float eta = conv(Iapix, hpix, length) / conv(Irpix, hpix, length);
    cout << "the value of eta is:"  << eta << endl;

    //Calculate the transmission
    Mat t(nr, nl, CV_32FC1);
    for (int i = 0; i < nr; i++)
    {
        float* iaPtr = Ia.ptr<float>(i);
        float* irPtr = Ir.ptr<float>(i);
        float* tPtr = t.ptr<float>(i);
        for (int j = 0; j < nl; j++)
        {
            *tPtr = 1 - (*iaPtr - eta * (*irPtr)) / normA;
            tPtr++; iaPtr++; irPtr++;
        }
    }
    imshow("t1", t);
    Mat trefined;
    trefined = stress(t);
    return trefined;
}

Mat stress(Mat& input)
{
    float data_max = 0.0, data_min = 5.0;
    int nr = input.rows; int nl = input.cols;
    Mat output(nr, nl, CV_32FC1);
    for (int i = 0; i < nr; i++)
    {
        float* data = input.ptr<float>(i);
        for (int j = 0; j < nl; j++)
        {
            if (*data > data_max) data_max = *data;
            if (*data < data_min) data_min = *data;
            data++;
        }
    }
    float temp = data_max - data_min;
    for (int i = 0; i < nr; i++)
    {
        float* indata = input.ptr<float>(i);
        float* outdata = output.ptr<float>(i);
        for (int j = 0; j < nl; j++)
        {
            *outdata = (*indata - data_min) / temp;
            if (*outdata < 0.1) *outdata = 0.1;
            indata++; outdata++;
        }
    }
    return output;
}