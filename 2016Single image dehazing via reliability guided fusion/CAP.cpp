#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <algorithm>
#include <random>
#include <opencv2\opencv.hpp>

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

Mat getDepthmap(Mat& input, float theta0, float theta1, float theta2);
Mat normrnd(float aver, float sigma, int row, int col);
Mat minFilter(Mat& input, int kernelSize);
Mat guidedfilter(Mat& srcImage, Mat& srcClone, int r, double eps);
Mat recover(Mat& srcimg, Mat& t, float *array);

int main() 
{
    string loc = "F:/JZYY/pic/dehazing/7w.jpg" ;
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
    Mat convertImage; 
    resizedImage.convertTo(convertImage, CV_32FC3, 1 / 255.0, 0);
    int kernelSize = 7; float eps = 0.0001;
    int radius = kernelSize / 2;

    Mat depmap(rows, cols, CV_32FC1);
    float the0 = 0.121779, the1 = 0.959710, the2 = -0.780245;
    float aveg = 0.0; float sigma = 0.041337;
    depmap = getDepthmap(convertImage, the0, the1, the2);
    Mat guassian = normrnd(aveg, sigma, rows, cols);
    depmap += guassian;

    imshow("depmap", depmap);

    Mat refdep = minFilter(depmap, kernelSize);
    Mat finaldep(rows, cols, CV_32FC1);
    Mat graymat(rows, cols, CV_8UC1);
    Mat graymat_32F(rows, cols, CV_32FC1);

    cvtColor(image, graymat, COLOR_BGR2GRAY);

    for (int i = 0; i < rows; i++)
    {
        const uchar* inData = graymat.ptr<uchar>(i);
        float* outData = graymat_32F.ptr<float>(i);
        for (int j = 0; j < cols; j++)
        {
            *outData++ = *inData++ / 255.0;
        }
    }

    float epsilon = 0.0001;
    finaldep = guidedfilter(image, refdep, 6 * kernelSize, epsilon);
    
    //estimate Airlight
    cout << "estimating airlight." << endl;
    rows = depmap.rows, cols = depmap.cols;
    int pixelTot = rows * cols * 0.001;
    int *A = new int[3];
    Pixel *toppixels, *allpixels;
    toppixels = new Pixel[pixelTot];
    allpixels = new Pixel[rows * cols];

    for (unsigned int r = 0; r < rows; r++) 
    {
        const uchar *data = depmap.ptr<uchar>(r);
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
    cout << "airlight estimated as: " << A[0] << ", " << A[1] << ", " << A[2] << endl;

    float tmp_A[3];
    tmp_A[0] = A[0] / 255.0;
    tmp_A[1] = A[1] / 255.0;
    tmp_A[2] = A[2] / 255.0;

    float beta = 1.2;     //0.6~1.8
    Mat trans;
    cv::exp(-beta * finaldep, trans);
    cout << "tansmission estimated." << endl;
    imshow("t", trans);

    cout << "start recovering." << endl;
    Mat finalImage = recover(convertImage, trans, tmp_A);
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
    imwrite("./images/7wCAP_1.2.png", finalImage);
    destroyAllWindows();
    image.release();
    resizedImage.release();
    convertImage.release();
    trans.release();
    finalImage.release();
    return 0;
}

Mat getDepthmap(Mat& input, float theta0, float theta1,float theta2)
{
    Mat Ihsv; Mat output; Mat depth;

    cv::cvtColor(input, Ihsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat>hsv_vec;
    cv::split(Ihsv, hsv_vec);
    cv::addWeighted(hsv_vec[1], theta2, hsv_vec[2], theta1, theta0, output);
    depth = output;

    return depth;
}

Mat normrnd(float aver, float sigma, int row, int col)
{
    Mat p(row, col, CV_32FC1);
    
    random_device rd;
    mt19937 gen(rd());
    
    for (int i = 0; i < row; i++)
    {
        float *pData = p.ptr<float>(i);
        for (int j = 0; j < col; j++)
        {
            normal_distribution<float> normal(aver, sigma);
            *pData = normal(gen);
            pData++;
        }
    }
    return p;
}
 
Mat minFilter(Mat& input, int kernelSize)
{
    int row = input.rows; int col = input.cols;
    int radius = kernelSize / 2;
    Mat parseImage;
    copyMakeBorder(input, parseImage, radius, radius, radius, radius, BORDER_REPLICATE);
    Mat output = Mat::zeros(input.rows, input.cols, CV_32FC1);

    for (unsigned int r = 0; r < row; r++)
    {
        float *fOutData = output.ptr<float>(r);

        for (unsigned int c = 0; c < col; c++)
        {
            Rect ROI(c, r, kernelSize, kernelSize);
            Mat imageROI = parseImage(ROI);
            double minValue = 0, maxValue = 0;
            Point minPt, maxPt;
            minMaxLoc(imageROI, &minValue, &maxValue, &minPt, &maxPt);

            *fOutData++ = minValue;
        }
    }

    return output;
}

Mat guidedfilter(Mat& srcImage, Mat& srcClone, int r, double eps)
{
    Mat graymat;
    cvtColor(srcImage, graymat, COLOR_BGR2GRAY);
    graymat.convertTo(srcImage, CV_32FC1, 1 / 255.0);
    //srcClone.convertTo(srcClone, CV_64FC1);
    int nRows = srcImage.rows;
    int nCols = srcImage.cols;
    Mat boxResult;

    boxFilter(Mat::ones(nRows, nCols, srcImage.type()), boxResult, CV_32FC1, Size(r, r));

    Mat mean_I;
    boxFilter(srcImage, mean_I, CV_32FC1, Size(r, r));

    Mat mean_p;
    boxFilter(srcClone, mean_p, CV_32FC1, Size(r, r));

    Mat mean_Ip;
    boxFilter(srcImage.mul(srcClone), mean_Ip, CV_32FC1, Size(r, r));
    Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

    Mat mean_II;
    boxFilter(srcImage.mul(srcImage), mean_II, CV_32FC1, Size(r, r));

    Mat var_I = mean_II - mean_I.mul(mean_I);
    Mat var_Ip = mean_Ip - mean_I.mul(mean_p);

    Mat a = cov_Ip / (var_I + eps);
    Mat b = mean_p - a.mul(mean_I);

    Mat mean_a;
    boxFilter(a, mean_a, CV_32FC1, Size(r, r));
    mean_a = mean_a / boxResult;
    Mat mean_b;
    boxFilter(b, mean_b, CV_32FC1, Size(r, r));
    mean_b = mean_b / boxResult;

    Mat resultMat = mean_a.mul(srcImage) + mean_b;
    return resultMat;
}

Mat recover(Mat& srcimg, Mat& t, float *array)
{
    int nr = srcimg.rows, nl = srcimg.cols;
    float tnow = t.at<float>(0, 0);
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

            if (tnow < 0.1)
            {
                tnow = 0.1;
            }
            else if (tnow > 0.9)
            {
                tnow = 0.9;
            }

            for (int i = 0; i < 3; i++) 
            {
                val = (*srcPtr++ - array[i]) / tnow + array[i];
                *outPtr++ = val + 10 / 255.0;
            }

        }
    }
    return finalimg;
}