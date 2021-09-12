#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <algorithm>
#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;




Mat minFilter(Mat srcImage, int kernelSize) 
{
    int radius = kernelSize / 2;
    int srcType = srcImage.type();
    int targetType = 0;

    if (srcType % 8 == 0) 
    {
        targetType = 0;
    }
    else 
    {
        targetType = 5;
    }

    Mat ret(srcImage.rows, srcImage.cols, targetType);
    Mat parseImage;
    copyMakeBorder(srcImage, parseImage, radius, radius, radius, radius, BORDER_REPLICATE);

    for (unsigned int r = 0; r < srcImage.rows; r++) 
    {
        float *fOutData = ret.ptr<float>(r);
        uchar *uOutData = ret.ptr<uchar>(r);
        for (unsigned int c = 0; c < srcImage.cols; c++) 
        {
            Rect ROI(c, r, kernelSize, kernelSize);
            Mat imageROI = parseImage(ROI);
            double minValue = 0, maxValue = 0;
            Point minPt, maxPt;
            minMaxLoc(imageROI, &minValue, &maxValue, &minPt, &maxPt);
            if (!targetType) 
            {
                *uOutData++ = (uchar)minValue;
                continue;
            }
            *fOutData++ = minValue;
        }
    }
    return ret;
}

Mat maxFilter(Mat srcImage, int kernelSize)
{
    int radius = kernelSize / 2;
    int srcType = srcImage.type();
    int targetType = 0;

    if (srcType % 8 == 0)
    {
        targetType = 0;
    }
    else
    {
        targetType = 5;
    }

    Mat ret(srcImage.rows, srcImage.cols, targetType);
    Mat parseImage;
    copyMakeBorder(srcImage, parseImage, radius, radius, radius, radius, BORDER_REPLICATE);

    for (unsigned int r = 0; r < srcImage.rows; r++)
    {
        float *fOutData = ret.ptr<float>(r);
        uchar *uOutData = ret.ptr<uchar>(r);
        for (unsigned int c = 0; c < srcImage.cols; c++)
        {
            Rect ROI(c, r, kernelSize, kernelSize);
            Mat imageROI = parseImage(ROI);
            double minValue = 0, maxValue = 0;
            Point minPt, maxPt;
            minMaxLoc(imageROI, &minValue, &maxValue, &minPt, &maxPt);
            if (!targetType)
            {
                *uOutData++ = (uchar)maxValue;
                continue;
            }
            *fOutData++ = maxValue;
        }
    }
    return ret;
}

void makeDepth32f(Mat& source, Mat& output)
{
    if ((source.depth() != CV_32F) > FLT_EPSILON)
        source.convertTo(output, CV_32F);
    else
        output = source;
}

void guidedFilter(Mat& source, Mat& guided_image, Mat& output, int radius, float epsilon)
{
    CV_Assert(radius >= 2 && epsilon > 0);
    CV_Assert(source.data != NULL && source.channels() == 1);
    CV_Assert(guided_image.channels() == 1);
    CV_Assert(source.rows == guided_image.rows && source.cols == guided_image.cols);

    Mat guided;
    if (guided_image.data == source.data)
    {
        //make a copy
        guided_image.copyTo(guided);
    }
    else
    {
        guided = guided_image;
    }

    //将输入扩展为32位浮点型，以便以后做乘法
    Mat source_32f, guided_32f;
    makeDepth32f(source, source_32f);
    makeDepth32f(guided, guided_32f);

    //计算I*p和I*I
    Mat mat_Ip, mat_I2;
    multiply(guided_32f, source_32f, mat_Ip);
    multiply(guided_32f, guided_32f, mat_I2);

    //计算各种均值
    Mat mean_p, mean_I, mean_Ip, mean_I2;
    Size win_size(2 * radius + 1, 2 * radius + 1);
    boxFilter(source_32f, mean_p, CV_32F, win_size);
    boxFilter(guided_32f, mean_I, CV_32F, win_size);
    boxFilter(mat_Ip, mean_Ip, CV_32F, win_size);
    boxFilter(mat_I2, mean_I2, CV_32F, win_size);

    //计算Ip的协方差和I的方差
    Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
    Mat var_I = mean_I2 - mean_I.mul(mean_I);
    var_I += epsilon;

    //求a和b
    Mat a, b;
    divide(cov_Ip, var_I, a);
    b = mean_p - a.mul(mean_I);

    //对包含像素i的所有a、b做平均
    Mat mean_a, mean_b;
    boxFilter(a, mean_a, CV_32F, win_size);
    boxFilter(b, mean_b, CV_32F, win_size);

    //计算输出 (depth == CV_32F)
    output = mean_a.mul(guided_32f) + mean_b;
}

Mat getTransmission(Mat& srcimg, Mat& transmission,  int windowsize)
{
    int nr = srcimg.rows, nl = srcimg.cols;
    Mat trans(nr, nl, CV_32FC1);
    Mat graymat(nr, nl, CV_8UC1);
    Mat graymat_32F(nr, nl, CV_32FC1);

    if (srcimg.type() % 8 != 0) {
        cvtColor(srcimg, graymat_32F, COLOR_BGR2GRAY);
        guidedFilter(transmission, graymat_32F, trans, 6 * windowsize, 0.001);
    }
    else {
        cvtColor(srcimg, graymat, COLOR_BGR2GRAY);

        for (int i = 0; i < nr; i++) {
            const uchar* inData = graymat.ptr<uchar>(i);
            float* outData = graymat_32F.ptr<float>(i);
            for (int j = 0; j < nl; j++)
                *outData++ = *inData++ / 255.0;
        }
        guidedFilter(transmission, graymat_32F, trans, 6 * windowsize, 0.001);
    }
    return trans;
}

Mat recover(Mat& srcimg, Mat& t, float *array, int windowsize)
{
    //J(x) = (I(x) - A) / max(t(x), t0) + A;
    int radius = windowsize / 2;
    int nr = srcimg.rows, nl = srcimg.cols;
    float tnow = t.at<float>(0, 0);
    float t0 = 0.1;
    Mat finalimg = Mat::zeros(nr, nl, CV_32FC3);
    float val = 0;

    for (unsigned int r = 0; r < nr; r++)
    {
        const float* transPtr = t.ptr<float>(r);
        const float* srcPtr = srcimg.ptr<float>(r);
        float* outPtr = finalimg.ptr<float>(r);
        for (unsigned int c = 0; c < nl; c++) 
        {
            tnow = *transPtr++;
            tnow = std::max(tnow, t0);
            for (int i = 0; i < 3; i++)
            {
                val = (*srcPtr++ - array[i]) / tnow + array[i];
                *outPtr++ = val + 10 / 255.0;
            }
        }
    }
    return finalimg;
}
