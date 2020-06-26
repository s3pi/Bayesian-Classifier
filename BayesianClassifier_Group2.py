import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def ContourPlot(cls_all,cmean,cov_mat,limi,caption,clsvar):
    colorUse = ("red", "green", "blue")
    catColor = (u'#FFAFAF', u'#BBFFB9', u'#BBB9FF')
    contourColor = {"Purples","Greens","Blues"}  
    groupName = ("Class1", "Class2", "Class3")
    markSym = ('o', '^', 's')
    plt.xlabel('Xaxis')
    plt.ylabel('Yaxis')
    plt.figure()
    cls_xmin,cls_xmax,cls_ymin,cls_ymax = [],[],[],[]
    for clsno in range(0,limi):
        xval = cls_all[clsno][0]
        yval = cls_all[clsno][1]
        xmax = np.max(xval)
        ymax = np.max(yval)
        xmin = np.min(xval)
        ymin = np.min(yval)
        cls_xmin.append(xmin)
        cls_ymin.append(ymin)
        cls_xmax.append(xmax)
        cls_ymax.append(ymax)
        dx = 0.005
        upts = np.arange(xmin, xmax, (xmax - xmin) * dx) #find all points from xmin to xmax in steps of 0.005
        vpts = np.arange(ymin, ymax, (ymax - ymin) * dx)
        xpts, ypts = np.meshgrid(upts, vpts) #build a mesh using the above points
        zpts = []
        for i in range(len(xpts)):
            zx = [] 
            for j in range(len(ypts)):
                gx = find_gix([xpts[i][j], ypts[i][j]], cmean, cov_mat, clsno)
                zx.append(gx)
            zpts.append(zx)
        plt.contour(xpts,ypts,zpts, 10)
    xmin1 = min(cls_xmin) - 1.5
    ymin1 = min(cls_ymin) - 1.5
    xmax1 = max(cls_xmax) + 1.5
    ymax1 = max(cls_ymax) + 1.5
    x0,y0,x1,y1,x2,y2 = [], [],[],[],[] ,[]
    for a in np.arange(xmin1, xmax1, (xmax1-xmin1)/200.0):
        for b in np.arange(ymin1, ymax1, (ymax1-ymin1)/200.0):
           whclass,gx = FindGixnDB([a,b],cmean,cov_mat,limi)
           if(whclass==0):
               x0.append(a)
               y0.append(b) 
           elif(whclass==1):
               x1.append(a)
               y1.append(b) 
           else: 
               x2.append(a)
               y2.append(b)
    
    plt.scatter(x0,y0,alpha=0.4,marker='s', edgecolors=catColor[0], facecolor=catColor[0], s=50)
    plt.scatter(x1,y1,alpha=0.4,marker='s', edgecolors=catColor[1], facecolor=catColor[1], s=50)
    plt.scatter(x2,y2,alpha=0.4,marker='s', edgecolors=catColor[2], facecolor=catColor[2], s=50)
    for clsno in range(0,limi):
        xval = cls_all[clsno][0]
        yval = cls_all[clsno][1]
        plt.scatter(xval,yval,alpha=1.0,marker=markSym[clsno],edgecolors=colorUse[clsno],facecolor=colorUse[clsno],s=15,label=clsvar[clsno])
    plt.legend()
    plt.xlabel('Xaxis')
    plt.ylabel('Yaxis')
    plt.savefig("Type"+str(caption[0])+"Case"+str(caption[1])+str(clsvar[0])+str(clsvar[1])+str(clsvar[2])+".png")
    
        
def FindGixnDB(Xi,Mui,cov_matrix,limit):
    gix = []
    iclass = []
    for lim in range(0,limit):
        x = np.subtract(Xi,Mui[lim])
        xt = x.T
        inv_covmat = np.linalg.inv(cov_matrix[lim])
        sigma = np.linalg.det(cov_matrix[lim])
        if sigma == 0:
            sigma = 0.0000001
            g1x =0
        gx = -(1 / 2.0) * np.matmul(np.matmul(xt, inv_covmat), x) - (1 / 2.0) * np.log(sigma) - np.log(2*np.pi)
        gix.append(gx)        
        iclass.append(lim)
    #print(gix, iclass)
    gix_class = zip(gix,iclass)
    retgx = max(gix)
    whclass = max(gix_class)[1]
    #print(whclass)
    return whclass,retgx
        
    
def find_gix(Xi,Mui,cov_matrix,clsval):
    x = np.subtract(Xi,Mui[clsval])
    xt = x.T
    inv_covmat = np.linalg.inv(cov_matrix[clsval])
    det_cov = np.linalg.det(cov_matrix[clsval])
    if det_cov == 0:
        det_cov = 0.0000001
    gix =0
    gix = -(1 / 2.0) * np.matmul(np.matmul(xt, inv_covmat), x) - (1 / 2.0) * np.log(det_cov) - np.log(2*np.pi)
    return gix
    
def main():
    #storing paths of the traning files
    LSFile1_tr = "/Users/3pi/Documents/Pattern Recognition/Latest/LS/train_class1.txt"
    LSFile2_tr = "/Users/3pi/Documents/Pattern Recognition/Latest/LS/train_class2.txt"
    LSFile3_tr = "/Users/3pi/Documents/Pattern Recognition/Latest/LS/train_class3.txt"
    NLSFile1_tr = "/Users/3pi/Documents/Pattern Recognition/Latest/NLS/train_class1.txt"
    NLSFile2_tr = "/Users/3pi/Documents/Pattern Recognition/Latest/NLS/train_class2.txt"
    NLSFile3_tr = "/Users/3pi/Documents/Pattern Recognition/Latest/NLS/train_class3.txt"
    RdFile1_tr = "/Users/3pi/Documents/Pattern Recognition/Latest/RD/train_class1.txt"
    RdFile2_tr = "/Users/3pi/Documents/Pattern Recognition/Latest/RD/train_class2.txt"
    RdFile3_tr = "/Users/3pi/Documents/Pattern Recognition/Latest/RD/train_class3.txt"
    
    #mannually enter choice and optcase
    choice = 2
    opt_case = "1"
    btwc = "4"
    print("type = ", choice, "case = ", opt_case)
    #store paths to all the training data
    filePtrs = [[LSFile1_tr,LSFile2_tr,LSFile3_tr],[NLSFile1_tr,NLSFile2_tr,NLSFile3_tr],[RdFile1_tr,RdFile2_tr,RdFile3_tr]]
    
    #store covariance matrices for all 3 classes
    cova1, cova2, cova3 = [], [], []

    #store the 2 components of feature vectors/data points in an 1d array and find 2 * 2 cov matrix for class 1.
    c1x, c1y = np.loadtxt(filePtrs[int(choice)-1][0], delimiter=' ', usecols=(0, 1), unpack=True)
    cova1 = np.cov(c1x,c1y)

    #store the 2 components of feature vectors/data points in an 1d array and find 2 * 2 cov matrix for class 2.
    c2x, c2y = np.loadtxt(filePtrs[int(choice)-1][1], delimiter=' ', usecols=(0, 1), unpack=True)
    cova2 = np.cov(c2x,c2y)

    #store the 2 components of feature vectors/data points in an 1d array and find 2 * 2 cov matrix for class 3.
    c3x, c3y = np.loadtxt(filePtrs[int(choice)-1][2], delimiter=' ', usecols=(0, 1), unpack=True)
    cova3 = np.cov(c3x,c3y)

    #Covariance matrix is modified for each case
    #cova_res stores cov matrices of all classes in each case
    cova_res = []
    if(opt_case == str(1)):
        cov1 = (cova1 + cova2 + cova3) / 3
        cov1[0][0] = (cov1[0][0] + cov1[1][1])/2
        cov1[0][1] = 0.0
        cov1[1][0] = 0.0
        cov1[1][1] = cov1[0][0]
        cova_res = [cov1, cov1, cov1]
    elif(opt_case==str(2)):
        cov1 = (cova1 + cova2 + cova3)/3
        cova_res = [cov1, cov1, cov1]
    elif(opt_case==str(3)):
        cova1[0][1] = 0.0
        cova1[1][0] = 0.0
        cova2[0][1] = 0.0
        cova2[1][0] = 0.0
        cova3[0][1] = 0.0
        cova3[1][0] = 0.0
        cova_res = [cova1, cova2, cova3]
    elif(opt_case==str(4)):
        cova_res = [cova1, cova2, cova3]
    
    #storing both components of the vectors of all classes
    c1 = [c1x, c1y]
    c2 = [c2x, c2y]
    c3 = [c3x, c3y]

    #find the mean of the both components of vectors for all classes. Its an array of 2 elements.
    c1mean = [c1x.mean(), c1y.mean()]
    c2mean = [c2x.mean(), c2y.mean()]
    c3mean = [c3x.mean(), c3y.mean()]

    '''
    limi = 0

    #printing between classes you want to choose
    
    btwCls = ['C1C2','C2C3','C1C3','C1C2C3'] 
    for i in range(0,4):
        print(str(i+1) + ":" + btwCls[i])
    '''

    cls_all, cls_mean, clsvar = [], [], []
    #cls_all is a 3d array, which stores data points of 2 cols of all 3 classes.
    #cls_mean is a 2d array, which stores mean of chosen classes.
    if(btwc == str(1)):
        cls_all = [c1, c2]
        cls_mean = [c1mean, c2mean]
        limi = 2
        clsvar = ['C1','C2','']
    elif(btwc==str(2)):
        cls_all = [c2,c3]
        cls_mean = [c2mean,c3mean]
        limi=2
        clsvar = ['C2','C3','']
    elif(btwc==str(3)):
        cls_all = [c1,c3]
        cls_mean = [c1mean,c3mean]
        limi=2
        clsvar = ['C1','C3','']
    elif(btwc==str(4)):
        cls_all = [c1,c2,c3]
        cls_mean = [c1mean,c2mean,c3mean]
        limi=3
        clsvar = ['C1','C2','C3']

    '''
    caption = []
    caption.append(int(choice))
    caption.append(int(opt_case))
    caption.append(int(btwc))
    ContourPlot(cls_all,cls_mean, cova_res, limi, caption, clsvar)
    '''
    
    confusion_matrix = compute_confusion_matrix(choice, cls_all, cls_mean, cova_res, 3)
    print("\nconfusion_matrix")
    print_matrix(confusion_matrix)

    performance_matrix = find_performance_matrix(confusion_matrix)
    print("\nperformance_matrix")
    print_matrix(performance_matrix)
    
    class_accuracy = find_accuracy(confusion_matrix)
    print("\nclass accuracy: ", class_accuracy)

def print_matrix(matrix):
    for i in range(len(matrix)):
        print(matrix[i])

def compute_confusion_matrix(choice, cls_all, cls_mean, cova_res, limi):
    #storing paths of the test files
    LSFile1_test = "/Users/3pi/Documents/Pattern Recognition/Latest/LS/test_class1.txt"
    LSFile2_test = "/Users/3pi/Documents/Pattern Recognition/Latest/LS/test_class2.txt"
    LSFile3_test = "/Users/3pi/Documents/Pattern Recognition/Latest/LS/test_class3.txt"
    NLSFile1_test = "/Users/3pi/Documents/Pattern Recognition/Latest/NLS/test_class1.txt"
    NLSFile2_test = "/Users/3pi/Documents/Pattern Recognition/Latest/NLS/test_class2.txt"
    NLSFile3_test = "/Users/3pi/Documents/Pattern Recognition/Latest/NLS/test_class3.txt"
    RdFile1_test = "/Users/3pi/Documents/Pattern Recognition/Latest/RD/test_class1.txt"
    RdFile2_test = "/Users/3pi/Documents/Pattern Recognition/Latest/RD/test_class2.txt"
    RdFile3_test = "/Users/3pi/Documents/Pattern Recognition/Latest/RD/test_class3.txt"

    filePtrs_test = [[LSFile1_test, LSFile2_test, LSFile3_test],[NLSFile1_test, NLSFile2_test, NLSFile3_test],[RdFile1_test, RdFile2_test, RdFile3_test]]
    c1x, c1y = np.loadtxt(filePtrs_test[int(choice)-1][0], delimiter=' ', usecols=(0, 1), unpack=True)

    c2x, c2y = np.loadtxt(filePtrs_test[int(choice)-1][1], delimiter=' ', usecols=(0, 1), unpack=True)

    c3x, c3y = np.loadtxt(filePtrs_test[int(choice)-1][2], delimiter=' ', usecols=(0, 1), unpack=True)

    confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    for i in range(len(c1x)):
        whclass, gx = FindGixnDB([c1x[i], c1y[i]], cls_mean, cova_res, limi)
        confusion_matrix[0][whclass] += 1

    for i in range(len(c2x)):
        whclass, gx = FindGixnDB([c2x[i], c2y[i]], cls_mean, cova_res, limi)
        confusion_matrix[1][whclass] += 1

    for i in range(len(c3x)):
        whclass, gx = FindGixnDB([c3x[i], c3y[i]], cls_mean, cova_res, limi)
        confusion_matrix[2][whclass] += 1

    return confusion_matrix

def find_performance_matrix(confusion_matrix):
    
    performance_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for class_num in range(3):
        performance_matrix[class_num][0] = find_precision(confusion_matrix, class_num)
    for class_num in range(3):
        performance_matrix[class_num][1] = find_recall_rate(confusion_matrix, class_num)
    for class_num in range(3):
        performance_matrix[class_num][2] = find_f_score(performance_matrix[class_num])

    #find mean precision, mean recall rate and mean f score
    for metric in range(3):
        performance_matrix[3][metric] = (performance_matrix[0][metric] + performance_matrix[1][metric] + performance_matrix[2][metric]) / 3

    round_off_performance_matrix(performance_matrix)

    return performance_matrix

def find_accuracy(confusion_matrix):
    total_samples = 0
    for class_num in range(3):
        for metric in range(3):
            total_samples += confusion_matrix[class_num][metric]

    class_accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[2][2]) / total_samples * 100

    return class_accuracy

def round_off_performance_matrix(performance_matrix):
    for i in range(len(performance_matrix)):
        for j in range(len(performance_matrix[i])):
            performance_matrix[i][j] = round(performance_matrix[i][j], 2)


def find_precision(confusion_matrix, class_num):
    total_samples_classfied_as_class_num = 0
    for i in range(3):
        total_samples_classfied_as_class_num += confusion_matrix[i][class_num]
    precision_rate = (confusion_matrix[class_num][class_num] / total_samples_classfied_as_class_num) * 100

    return precision_rate

def find_recall_rate(confusion_matrix, class_num):
    total_samples_in_class = 0
    for j in range(3):
        total_samples_in_class += confusion_matrix[class_num][j]
    recall_rate = (confusion_matrix[class_num][class_num] / total_samples_in_class) * 100

    return recall_rate

def find_f_score(array):
    precision = array[0]
    recall = array[1]
    f_score = (precision * recall) / ((precision + recall) / 2)

    return f_score


main()


