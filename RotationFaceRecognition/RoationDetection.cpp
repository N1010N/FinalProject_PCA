#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <windows.h>
#include <vector>


#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

/* first of all you should set this configuration
	1.textfilepath which is path of your textfile that contains dataset information
	2.input & output path
	3.numClassification
	4.numComponent for PCA*/

string textFilePath = "F:\\nastaran\\university\\Proje payani\\RotationFaceRecognition\\Resources";
string input = "F:\\nastaran\\university\\Proje payani\\RotationFaceRecognition\\Resources\\Dataset1";
string output = "F:\\nastaran\\university\\Proje payani\\RotationFaceRecognition\\Output\\resultDataSet1";
string txtFilename = "Dataset1ImageInformation.txt";
int numClassification = 11;
ofstream out;
int numComponent = 100;


//***************************************************functions*******************************************
void save(const string &file_name, cv::PCA pca_)
{
	FileStorage fs(file_name, FileStorage::WRITE);
	fs << "mean" << pca_.mean;
	fs << "e_vectors" << pca_.eigenvectors;
	fs << "e_values" << pca_.eigenvalues;
	fs.release();
}


void load(const string &file_name, cv::PCA &pca_)
{
	FileStorage fs(file_name, FileStorage::READ);
	fs["mean"] >> pca_.mean;
	fs["e_vectors"] >> pca_.eigenvectors;
	fs["e_values"] >> pca_.eigenvalues;
	fs.release();

}
//***************************************************************************************************
static Mat norm_0_255(InputArray _src) {


	Mat src = _src.getMat();
	// Create and return normalized image:
	Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);

		break;
	case 3:

		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}

	return dst;

}

//************************************************************

static void writeIntxtFile(int classLabel)// class label is name of folder in dataset too
{
	
	WIN32_FIND_DATA fd;
	ostringstream convert;
	convert << classLabel;
	string classlabelstring;
	classlabelstring = convert.str();
	string str = input + "/" + classlabelstring + "/" + "*.jpg";
	string pathImages = input + "/" + classlabelstring + "/";
	char currentPath[500];
	strcpy(currentPath, str.c_str());
	HANDLE hFind = ::FindFirstFile(currentPath, &fd);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			// read all (real) files in current folder
			// , delete '!' read other 2 default folder . and ..
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {

				out << pathImages << fd.cFileName << ";"<< classLabel << endl;
				//names.push_back(fd.cFileName);
			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
	
	
	
}
//*******************************************************\readcv/*******************

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    ifstream file(textFilePath+"//"+filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);

		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}
//*******************************************************************************
// Converts the images given in src into a row matrix.
Mat asRowMatrix(const vector<Mat>& src, int rtype, double alpha = 1, double beta = 0) {

	// Number of samples:
	size_t n = src.size();
	// Return empty matrix if no matrices given:
	if (n == 0)
		return Mat();
	// dimensionality of (reshaped) samples
	size_t d = src[0].total();
	// Create resulting data matrix:
	Mat data(n, d, rtype);
	// Now copy data:
	for (int i = 0; i < n; i++) {
		//
		if (src[i].empty()) {
			string error_message = format("Image number %d was empty, please check your input data.", i);
			CV_Error(CV_StsBadArg, error_message);
		}
		// Make sure data can be reshaped, throw a meaningful exception if not!
		if (src[i].total() != d) {
			string error_message = format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, d, src[i].total());
			CV_Error(CV_StsBadArg, error_message);
		}
		// Get a hold of the current row:
		Mat xi = data.row(i);
		// Make reshape happy by cloning for non-continuous matrices:
		if (src[i].isContinuous()) {
			src[i].reshape(1, 1).convertTo(xi, rtype, alpha, beta);
		}
		else {
			src[i].clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
		}
	}
	return data;
}
//####################################################\  MAIN /##########################################
int main()
{

	CvSVM svm;
	PCA pca;

	if (!std::ifstream("RotationDetection(svm)"))
	{
		out.open(textFilePath + "\\" + txtFilename, ios::in);// we want to create a textfile of names
		vector<Mat> images;
		vector<int> labels;
	try{

		/*for (int i = 1; i <= numClassification; i++)
		{
			writeIntxtFile(i);
		}*/

		read_csv(txtFilename, images, labels);

	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << txtFilename << "\". Reason: " << e.msg << endl;
		// nothing more we can do
		exit(1);
	}

	// Quit if there are not enough images for this demo.
	if (images.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(CV_StsError, error_message);
	}


	//********************************************************    Create PCA         **************************

	Mat data = asRowMatrix(images, CV_32FC1);
	pca=PCA(data, Mat(), CV_PCA_DATA_AS_ROW, numComponent);

	// And copy the PCA results:
	Mat mean = pca.mean.clone();
	Mat eigenvalues = pca.eigenvalues.clone();
	Mat eigenvectors = pca.eigenvectors.clone();

	int k = eigenvalues.rows;
	int p = labels.size();

	Mat result = Mat(p, k, CV_32FC1);
	pca.project(data, result);// coefficient matrices

	imwrite(format("%s/mean.png", output.c_str()), norm_0_255(mean.reshape(1, images[0].rows)));


	int height = images[0].rows;

	for (int i = 0; i < k; i++)
	{
		Mat ev = eigenvectors.row(i).clone();
		// Reshape to original size & normalize to [0...255] for imshow.
		Mat grayscale = norm_0_255(ev.reshape(1, height));
		// Show the image & apply a Jet colormap for better sensing.
		Mat cgrayscale;
		applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
		imwrite(format("%s/eigenface_%d.png", output.c_str(), i), norm_0_255(cgrayscale));
	}

	int n = labels.size();
	//int *arrayLabel = new int[n + 2];
	Mat labelsCol=Mat (n, 1, CV_32SC1);//type of labels must be float

	memcpy(labelsCol.data, labels.data(), labels.size()*sizeof(int));


	//**********************************************************\ train by SVM /**********************************

	CvSVMParams params;
	params.svm_type = SVM::C_SVC;
	params.C = 0.1;
	params.kernel_type = SVM::LINEAR;
	params.term_crit = TermCriteria(CV_TERMCRIT_ITER, (int)1e7, 1e-6);
	// Train the SVM


	svm.train_auto(result, labelsCol, Mat(), Mat(), params);

	cout<< "train compeleted" << endl;

	svm.save("RotationDetection(svm)");
	cout << "train saved" << endl;
	save("pca", pca);

	}

	else
	{

		//CvSVM svm;
		cout << "File already exists" << std::endl;
		svm.load("RotationDetection(svm)");
		load("pca", pca);
		//return false;
	
	}


	//********************************************************************** \ test training  /************************

	string testpath = "F:\\nastaran\\university\\Proje payani\\RotationFaceRecognition\\Resources\\TestSample";
	string testResult = "testResult.txt";
	
	int numCorrect = 0, numWrong = 0;

	CvSVMParams params;
	params = svm.get_params();
	ofstream write(testpath+"//"+testResult,ios::app);
	write << "******************************************************" << endl;
	write << "Classification :" << numClassification << endl;
	write << " Component :" << numComponent << endl;
	write << " Auto train Parameters :" << endl;
	write << "svm_type: " << params.svm_type << endl;
	write << "C : " << params.C << endl;
	write << "kernel_type : " << params.kernel_type << endl;
	write << "term_crit eps: " << params.term_crit.epsilon<< endl;
	write << "term_crit max_Iter: " << params.term_crit.max_iter << endl;
	write << "term_crit type: " << params.term_crit.type << endl;

	ifstream labelsDoc("labels.txt",ios::in);
	vector <int> testLabels;
	int index = 0;
	while (!labelsDoc.eof())
	{
		labelsDoc >> index;
		testLabels.push_back(index);
	}

	Mat testSample;
	WIN32_FIND_DATA fd;
	
	string str = testpath +"\\"+"*.jpg";
	char currentPath[500];
	strcpy(currentPath, str.c_str());
	str = testpath + "\\";


    index = 0;
	HANDLE hFind = ::FindFirstFile(currentPath, &fd);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			// read all (real) files in current folder
			// , delete '!' read other 2 default folder . and ..
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {

				testSample=imread(str+fd.cFileName,0);

			//	cout << "testSample before is" << testSample.depth() << " " << testSample.channels() << endl;
				rsize_t total = testSample.total();
				


				Mat newTest = Mat(1, total, CV_32FC1);
				
				
				if (testSample.isContinuous()) {
					testSample.reshape(1, 1).convertTo(newTest, CV_32FC1, 1, 0);
				}
				else {
					testSample.clone().reshape(1, 1).convertTo(newTest, CV_32FC1, 1, 0);
				}

				memcpy(newTest.data, testSample.data, testSample.cols * sizeof(int));
				//pca(newTest, Mat(), CV_PCA_DATA_AS_ROW, numComponent);
				//Mat mean = pca.mean.clone();
				Mat eigenvalues1 = pca.eigenvalues.clone();
				//Mat eigenvectors2 = pca.eigenvectors.clone();

				int k = eigenvalues1.rows;
				//	int p = labels.size();

				Mat result1 = Mat(1,k, CV_32FC1);// testttt
				pca.project(newTest, result1);
				cout << "testSample after is" << testSample.depth() << " " << testSample.channels() << endl;


				//testSample=norm_0_255(testSample);
			///	testSample.c
				int predictLabel = svm.predict(result1);

				if (predictLabel == testLabels[index])
					numCorrect++;
				else
					numWrong++;

				index++;

				ostringstream convert;
				convert << predictLabel;
				write << str + fd.cFileName + ";" + convert.str()<<endl;
				//names.push_back(fd.cFileName);
			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
	write << "NumCorrect :" << numCorrect << endl;
	write << "NumWrong :" << numWrong << endl;
	write << "Total Sample :" << index  << endl;

	out.close();

	return 0;
}