#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

Mat minFilter(Mat& input, int kernelSize);
Mat boundCon(Mat& srcimg, float *airlight, int C0, int C1);
Mat calTrans(Mat& srcimg, Mat& t, float lambda, float param);
Mat calWeight(Mat& srcimg, Mat& kernel, float param);
Mat fft2(Mat I, Size size);
void ifft2(const Mat &src, Mat &Fourier);
void psf2otf(Mat& src, int rows, int cols, Mat& dst);
void circshift(Mat& img, int dw, int dh, Mat& dst);
Mat sign(Mat& input); Mat fliplr(Mat& input); Mat flipud(Mat& input);
Mat recover(Mat& srcimg, Mat& t, float *airlight, float delta);

int main()
{
    clock_t start, finish;
    double duration;

    cout << "A defog program" << endl
        << "----------------" << endl;

    Mat image = imread("F:/JZYY/pic/dehazing/4w.png"); 
    
    imshow("hazyiamge", image);
    cout << "input hazy image" << endl;

    Mat resizedImage;
    int originRows = image.rows;
    int originCols = image.cols;

    double scale = 1.0;

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

    vector<Mat> channels(3);
    split(convertImage, channels);

    Mat R = channels[2];
    Mat G = channels[1];
    Mat B = channels[0];

    int kernelSize = 15;
    Mat minImgR = minFilter(channels[2], kernelSize);
    Mat minImgG = minFilter(channels[1], kernelSize);
    Mat minImgB = minFilter(channels[0], kernelSize);

    double RminValue = 0, RmaxValue = 0;
    Point RminPt, RmaxPt;
    minMaxLoc(minImgR, &RminValue, &RmaxValue, &RminPt, &RmaxPt);
    cout << RmaxValue << endl;

    double GminValue = 0, GmaxValue = 0;
    Point GminPt, GmaxPt;
    minMaxLoc(minImgG, &GminValue, &GmaxValue, &GminPt, &GmaxPt);
    cout << GmaxValue << endl;

    double BminValue = 0, BmaxValue = 0;
    Point BminPt, BmaxPt;
    minMaxLoc(minImgB, &BminValue, &BmaxValue, &BminPt, &BmaxPt);
    cout << BmaxValue << endl;

    float A[3];
    A[0] = BmaxValue; A[1] = GmaxValue; A[2] = RmaxValue;

    Mat trans = boundCon(convertImage, A, 30, 300);
    imshow("t", trans);

    int s = 3;
    Mat element; Mat transrefine;
    element = getStructuringElement(MORPH_ELLIPSE, Size(s, s));
    morphologyEx(trans, transrefine, MORPH_OPEN, element);
    imshow("transrefine", transrefine);
    
    float lambda = 2.0; float param = 0.5;
     Mat finaltrans = calTrans(convertImage, transrefine, lambda, param);
    imshow("finaltrans", finaltrans);

    float delta = 0.85;
    Mat recoverimg = recover(convertImage, finaltrans, A, delta);
    imshow("recover", recoverimg);

    recoverimg.convertTo(recoverimg, CV_8UC3, 255.0, 0);
    imwrite("./images/4wBCCR.png", recoverimg);

    finish = clock();
    duration = (double)(finish - start);
    cout << "defog used " << duration << "ms time;" << endl;
    waitKey(0);
    
    return 0;
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

Mat boundCon(Mat& srcimg, float *airlight, int C0, int C1)
{
    int nr = srcimg.rows; int nl = srcimg.cols;
    float c0 = C0 / 255.0; float c1 = C1 / 255.0;
    Mat trans = Mat::zeros(nr, nl, CV_32FC1);

    float airlight0 = airlight[0]; float airlight1 = airlight[1]; float airlight2 = airlight[2];

    for (unsigned int r = 0; r < nr; r++)
    {
        float* srcPtr = srcimg.ptr<float>(r);
        float* tPtr = trans.ptr<float>(r);
        float B, G, R; float t_b, t_g, t_r; 
        float tb; float tmax = 1.0;

        for (unsigned int c = 0; c < nl; c++) 
        {
            B = *srcPtr++; G = *srcPtr++; R = *srcPtr++;

            t_b = std::max((airlight0 - B) / (airlight0 - c0), (B - airlight0) / (c1 - airlight0));
            t_g = std::max((airlight1 - G) / (airlight1 - c0), (G - airlight1) / (c1 - airlight1));
            t_r = std::max((airlight2 - R) / (airlight2 - c0), (R - airlight2) / (c1 - airlight2));

            tb = t_b > t_g ? t_b : t_g;
            tb = tb > t_r ? tb : t_r;
            tb = std::min(tb, tmax);

            *tPtr++ = tb;
        }
    }
    return trans;
}

Mat calTrans(Mat& srcimg, Mat& t, float lambda, float param)
{
    int nsz = 3; int NUM = nsz * nsz;
    int nr = t.rows; int nl = t.cols;
    Size size = t.size();

    Mat kernel1 = (Mat_<float>(3, 3) << 5, 5, 5, -3, 0, -3, -3, -3, -3);
    Mat kernel2 = (Mat_<float>(3, 3) << -3, 5, 5, -3, 0, 5, -3, -3, -3);
    Mat kernel3 = (Mat_<float>(3, 3) << -3, -3, 5, -3, 0, 5, -3, -3, 5);
    Mat kernel4 = (Mat_<float>(3, 3) << -3, -3, -3, -3, 0, 5, -3, 5, 5);
    Mat kernel5 = (Mat_<float>(3, 3) << 5, 5, 5, -3, 0, -3, -3, -3, -3);
    Mat kernel6 = (Mat_<float>(3, 3) << -3, -3, -3, 5, 0, -3, 5, 5, -3);
    Mat kernel7 = (Mat_<float>(3, 3) << 5, -3, -3, 5, 0, -3, 5, -3, -3);
    Mat kernel8 = (Mat_<float>(3, 3) << 5, 5, -3, 5, 0, -3, -3, -3, -3);

    normalize(kernel1, kernel1, 1.0, 0.0, NORM_L2); normalize(kernel2, kernel2, 1.0, 0.0, NORM_L2);
    normalize(kernel3, kernel3, 1.0, 0.0, NORM_L2); normalize(kernel4, kernel4, 1.0, 0.0, NORM_L2);
    normalize(kernel5, kernel5, 1.0, 0.0, NORM_L2); normalize(kernel6, kernel6, 1.0, 0.0, NORM_L2);
    normalize(kernel7, kernel7, 1.0, 0.0, NORM_L2); normalize(kernel8, kernel8, 1.0, 0.0, NORM_L2);

    Mat d = Mat::zeros(3, 3, CV_32FC(8));
    vector<Mat> dchannels(8);
    split(d, dchannels);
    dchannels[0] = kernel1; dchannels[1] = kernel2; dchannels[2] = kernel3; dchannels[3] = kernel4;
    dchannels[4] = kernel5; dchannels[5] = kernel6; dchannels[6] = kernel7; dchannels[7] = kernel8;

    Mat wfun = Mat::zeros(nr, nl, CV_32FC(8));
    vector<Mat> wfunchannels(8);

    for (int k = 0; k < 8; k++)
    {
        wfunchannels[k] = calWeight(srcimg, dchannels[k], param);
    }

    Mat Tf;
    Tf=fft2(t,size);

    Mat D = Mat::zeros(nr, nl, CV_32FC(8));
    Mat DS = Mat::zeros(nr, nl, CV_32FC1);

    vector<Mat> Dchannels(8);
    split(D, Dchannels);

    for (int k = 0; k < 8; k++)
    {
        psf2otf(dchannels[k], nr, nl, Dchannels[k]);   
    
        Mat Dchannels_temp = Mat::zeros(nr, nl, CV_32FC1);
        vector<Mat> Dchannels_temp1(2);
        split(Dchannels[k], Dchannels_temp1);

        magnitude(Dchannels_temp1[0], Dchannels_temp1[1], Dchannels_temp);
        pow(Dchannels_temp, 2, Dchannels_temp);

        DS += Dchannels_temp;
    }

    float beta = 1.0; float beta_rate = 2 * sqrt(2);
    float beta_max = 256;                             

    while (beta < beta_max)
    {
        float gamma = lambda / beta;

        Mat dt = Mat::zeros(nr, nl, CV_32FC(8));
        Mat u = Mat::zeros(nr, nl, CV_32FC(8));
        Mat DU = Mat::zeros(nr, nl, CV_32FC2);

        vector<Mat> dtchannels(8);
        split(dt, dtchannels);

        vector<Mat> uchannels(8);
        split(u, uchannels);

        for (int k = 0; k < 8; k++)
        {
            float zero = 0.0;
            filter2D(t, dtchannels[k], t.depth(), dchannels[k]);
            uchannels[k] = max(abs(dtchannels[k]) - (1 / (8 * beta)) * wfunchannels[k], zero).mul(sign(dtchannels[k]));

            Mat dfliplr = fliplr(dchannels[k]); Mat dflipud = flipud(dfliplr);
            filter2D(uchannels[k], uchannels[k], uchannels[k].depth(), dflipud);

            Mat DUtemp;
            DUtemp=fft2(uchannels[k],size);
            DU += DUtemp;
        }

        Mat ifft;
        Mat tfftup = gamma * Tf + DU;
        Mat tfftdown = gamma * Mat::ones(nr, nl, CV_32FC1) + DS;

        vector<Mat> tfft_temp(2);
        split(tfftup, tfft_temp);
        tfft_temp[0] = tfft_temp[0] / tfftdown;
        tfft_temp[1] = tfft_temp[1] / tfftdown;

        Mat final_tfft;
        merge(tfft_temp, final_tfft);
        ifft2(final_tfft, ifft);

        t = abs(ifft);
        beta = beta * beta_rate;
    }

    Mat trans(nr, nl, CV_32FC1);
    trans = t;
    return trans;
}

Mat calWeight(Mat& srcimg, Mat& kernel, float param)
{
    int nr = srcimg.rows; int nl = srcimg.cols;
    Mat wfun; float weight = -1 / (param * 2);

    vector<Mat> channels(3);
    split(srcimg, channels);

    Mat d_b; Mat d_g; Mat d_r;
    filter2D(channels[0], d_b, channels[0].depth(), kernel);
    filter2D(channels[1], d_g, channels[1].depth(), kernel);
    filter2D(channels[2], d_r, channels[2].depth(), kernel);

    exp(weight * (d_b.mul(d_b) + d_g.mul(d_g) + d_r.mul(d_r)), wfun);

    return wfun;
}

Mat fft2(Mat I, Size size)
{
    Mat If = Mat::zeros(I.size(), I.type());

    Size dftSize;

    // compute the size of DFT transform
    dftSize.width = getOptimalDFTSize(size.width);
    dftSize.height = getOptimalDFTSize(size.height);

    // allocate temporary buffers and initialize them with 0's
    Mat tempI(dftSize, I.type(), Scalar::all(0));

    //copy I to the top-left corners of temp
    Mat roiI(tempI, Rect(0, 0, I.cols, I.rows));
    I.copyTo(roiI);

    if (I.channels() == 1)
    {
        dft(tempI, If, DFT_COMPLEX_OUTPUT);
    }
    else
    {
        vector<Mat> channels;
        split(tempI, channels);
        for (int n = 0; n<I.channels(); n++)
        {
            dft(channels[n], channels[n], DFT_COMPLEX_OUTPUT);
        }

        cv::merge(channels, If);
    }

    return If(Range(0, size.height), Range(0, size.width));
}

void ifft2(const Mat &src, Mat &Fourier)
{
    int mat_type = src.type();
    assert(mat_type < 15); 
    if (mat_type < 7)
    {
        Mat planes[] = { Mat_<double>(src), Mat::zeros(src.size(), CV_64F) };
        merge(planes, 2, Fourier);
        dft(Fourier, Fourier, DFT_INVERSE + DFT_SCALE, 0);
    }
    else 
    {
        Mat tmp;
        dft(src, tmp, DFT_INVERSE + DFT_SCALE, 0);
        vector<Mat> planes;
        split(tmp, planes);
        magnitude(planes[0], planes[1], planes[0]); 
        Fourier = planes[0];
    }
}

void psf2otf(Mat& src, int rows, int cols, Mat& dst)
{
    Mat src_fill = Mat::zeros(rows, cols, CV_32FC1);
    Mat src_fill_out = Mat::zeros(rows, cols, CV_32FC1);

    for (int i = 0; i < src.rows; i++)
    {
        float* data = src_fill.ptr<float>(i);
        float* data2 = src.ptr<float>(i);
        for (int j = 0; j < src.cols; j++)
        {
            data[j] = data2[j];
        }
    }
    
    Size size; size.height = rows; size.width = cols;
    circshift(src_fill, -int(src.rows / 2), -int(src.cols / 2), src_fill_out);
    dst = fft2(src_fill_out, size);
    
    return;
}

void circshift(Mat& img, int dw, int dh, Mat& dst)
{
    int rows = img.rows;
    int cols = img.cols;
    dst = img.clone();
    if (dw < 0 && dh < 0)
    {
        for (int i = 0; i < rows; i++)
        {
            int t = i + dw;
            if (t >= 0)
            {
                float* data = img.ptr<float>(i);
                float* data2 = dst.ptr<float>(t);
                for (int j = 0; j < cols; j++)
                {
                    data2[j] = data[j];
                }
            }
            else
            {
                float* data = img.ptr<float>(i);
                float* data2 = dst.ptr<float>(dst.rows + t);
                for (int j = 0; j < cols; j++)
                {
                    data2[j] = data[j];
                }
            }
        }


        for (int j = 0; j < cols; j++)
        {
            int t = j + dh;
            if (t >= 0)
            {
                for (int i = 0; i < rows; i++)
                {
                    float* data = img.ptr<float>(i);
                    float* data2 = dst.ptr<float>(i);
                    data2[t] = data[j];
                }
            }
            else
            {
                for (int i = 0; i < rows; i++)
                {
                    float* data = img.ptr<float>(i);
                    float* data2 = dst.ptr<float>(i);
                    data2[dst.cols + t] = data[j];
                }
            }
        }
    }
    return;
}

Mat sign(Mat& input)
{
    int nr = input.rows; int nl = input.cols;
    Mat output(nr, nl, CV_32FC1);

    for (int i = 0; i < nr; i++)
    {
        const float* inData = input.ptr<float>(i);
        float* outData = output.ptr<float>(i);
        for (int j = 0; j < nl; j++)
        {
            if (*inData > 0)
            {
                *outData = 1;
            }
            else if (*inData < 0)
            {
                *outData = -1;
            }
            else
            {
                *outData = 0;
            }
            outData++;
        }
    }
    return output;
}

Mat fliplr(Mat& input)
{
    int nr = input.rows; int nl = input.cols;
    Mat output(nr, nl, CV_32FC1); 
    float temp;

    for (int i = 0; i < nr; i++)
    {
        for (int j = 0; j < (nl - 1) / 2 + 1; j++)
        {
            temp = input.at<float>(i, j);
            output.at<float>(i, j) = input.at<float>(i, nl - j - 1);
            output.at<float>(i, nl - j - 1) = temp;
        }
    }
    return output;
}

Mat flipud(Mat& input)
{
    int nr = input.rows; int nl = input.cols;
    Mat output(nr, nl, CV_32FC1);
    float temp;

    for (int i = 0; i < (nr - 1) / 2 + 1; i++)
    {
        for (int j = 0; j < nl; j++)
        {
            temp = input.at<float>(i, j);
            output.at<float>(i, j) = input.at<float>(nr - 1 - i, j);
            output.at<float>(nr - 1 - i,j) = temp;
        }
    }
    return output;
}

Mat recover(Mat& srcimg, Mat& t, float *airlight, float delta)
{
    int nr = srcimg.rows, nl = srcimg.cols;
    float tnow = t.at<float>(0, 0);
    float t0 = 0.0001;

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
            pow(tnow, delta);
            for (int i = 0; i < 3; i++)
            {
                val = (*srcPtr++ - airlight[i]) / tnow + airlight[i];
                *outPtr++ = val;
            }
        }
    }
    return finalimg;
}