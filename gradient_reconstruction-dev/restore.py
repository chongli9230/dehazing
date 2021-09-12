import numpy as np
from skimage import filters
from skimage import io
from skimage.transform import rescale
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve
import matplotlib.pyplot as plt
import os

if __name__ == "__main__" :
    os.chdir(os.path.dirname(__file__))
    
    img_f = io.imread("data/cpma.jpg", as_gray=True)
    img_f = rescale(img_f, 0.5)
    (img_y, img_x) = img_f.shape
    
    #img_f = np.array([[1, 2, 3], [2, 2, 2], [3, 2, 1]])
    #img_x = 3
    #img_y = 3
    
    img = np.zeros((img_y + 2, img_x + 2))
    img[1:-1, 1:-1] = img_f
    
    #building gradient vectors
    grad_x = filters.sobel_h(img)
    grad_y = filters.sobel_v(img)
    
    grad_hh = filters.sobel_h(grad_x)
    grad_vv = filters.sobel_v(grad_y)
    
    #lapl = np.zeros((img_y, img_x))
    
    lapl = grad_hh + grad_vv
    #building divergence from gradient
    #lapl = filters.laplace(img)
    
    #A = np.zeros(((img_x + 2) * (img_y + 2), (img_x + 2) * (img_y + 2)))
    #b = np.zeros((img_x + 2) * (img_y + 2))
    
    A = np.zeros(((img_x) * (img_y), (img_x) * (img_y)))
    b = np.zeros((img_x) * (img_y))
    
    #init matrix for linear equation
    last_row = 0
    
    for y in range(img_y) :
        for x in range(img_x) :
            #print(str(x) + "; " + str(y))
            #print("Set (" + str(x) + "; " + str(y) + ") to -4.0")
            A[last_row, y * (img_x) + x] = -4.0
            if y - 1 >= 0 :
                #print("(top)Set (" + str(x) + "; " + str(y - 1) + ") to 1.0")
                A[last_row, (y - 1) * (img_x) + x] = 1.0
            if y + 1 <= img_y - 1 :
                #print("(bottom)Set (" + str(x) + "; " + str(y + 1) + ") to 1.0")
                A[last_row, (y + 1) * (img_x) + x] = 1.0
            if x - 1 >= 0 :
                #print("(left)Set (" + str(x - 1) + "; " + str(y) + ") to 1.0")
                A[last_row, y * (img_x) + x - 1] = 1.0
            if x + 1 <= img_x -1 :
                #print("(right)Set (" + str(x+1) + "; " + str(y) + ") to 1.0")
                A[last_row, y * (img_x) + x + 1] = 1.0
            b[last_row] = lapl[y + 1, x + 1]
            
            last_row += 1
            
    '''
    for y in range(img_y + 2) :
        for x in range(img_x + 2) :
            #print(str(x) + "; " + str(y))
            if x >= 1 and x <= img_x and y >= 1 and y <= img_y :
                #print("Set (" + str(x) + "; " + str(y) + ") to -4.0")
                A[last_row, y * (img_x + 2) + x] = -4.0
                if y - 1 >= 0 :
                    #print("(top)Set (" + str(x) + "; " + str(y - 1) + ") to 1.0")
                    A[last_row, (y - 1) * (img_x + 2) + x] = 1.0
                if y + 1 <= img_y + 1 :
                    #print("(bottom)Set (" + str(x) + "; " + str(y + 1) + ") to 1.0")
                    A[last_row, (y + 1) * (img_x + 2) + x] = 1.0
                if x - 1 >= 0 :
                    #print("(left)Set (" + str(x - 1) + "; " + str(y) + ") to 1.0")
                    A[last_row, y * (img_x + 2) + x - 1] = 1.0
                if x + 1 <= img_x + 1 :
                    #print("(right)Set (" + str(x+1) + "; " + str(y) + ") to 1.0")
                    A[last_row, y * (img_x + 2) + x + 1] = 1.0
                b[last_row] = -lapl[y, x]
            elif x == 0 :
                A[last_row, y * (img_x + 2) + x] = 1.0
                b[last_row] = 0.0
            elif y == 0 :
                A[last_row, y * (img_x + 2) + x] = 1.0
                b[last_row] = 0.0
            elif x == img_x + 1 :
                A[last_row, y * (img_x + 2) + x] = 1.0
                b[last_row] = 0.0
            elif y == img_y + 1 :
                A[last_row, y * (img_x + 2) + x] = 1.0
                b[last_row] = 0.0
            last_row += 1
    '''
    
    #print(str(A))
    #result = spsolve(A, b)
    result = solve(A, b)
    result = result.reshape((img_y, img_x))
    
    plt.figure(1)
    plt.subplot(211)
    plt.imshow(img_f)
    plt.subplot(212)
    plt.imshow(result)
    plt.show()
    #plt.imshow(result)
    #plt.show()