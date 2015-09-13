#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;




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

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
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


int main(int argc, const char *argv[]) {
	// Check for valid command line arguments, print usage
	// if no arguments were given.
	if (argc < 2) {
		cout << "usage: " << argv[0] << " <csv.ext> <output_folder> " << endl;
		exit(1);
	}
	string output_folder = ".";
	if (argc == 3) {
		output_folder = string(argv[2]);
	}
	// Get the path to your CSV.
	string fn_csv = string(argv[1]);
	cout << "filename: " << fn_csv << endl;

	// These vectors hold the images and corresponding labels.
	vector<Mat> images;
	vector<int> labels;
	// Read in the data. This can fail if no valid
	// input filename is given.
	try {
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// nothing more we can do
		exit(1);
	}
	// Quit if there are not enough images for this demo.
	if (images.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(CV_StsError, error_message);
	}
	// Get the height from the first image. We'll need this
	// later in code to reshape the images to their original
	// size:
	int height = images[0].rows;

	int a;
	
	Mat testSample = images[images.size() - 1];
	int testLabel = labels[labels.size() - 1];

	rsize_t total = testSample.total();
	Mat newTest = Mat(1, total, CV_32FC1);
	if (testSample.isContinuous()) {
		testSample.reshape(1, 1).convertTo(newTest, CV_32FC1, 1, 0);
	}
	else {
		testSample.clone().reshape(1, 1).convertTo(newTest, CV_32FC1, 1, 0);
	}


	images.pop_back();
	labels.pop_back();

	
	

	size_t n = labels.size();
	Mat labelsMat = Mat(n, 1, CV_32SC1);
	cout << labelsMat.rows<< labelsMat.cols << endl;
	
	//copy vector to mat  
	memcpy(labelsMat.data, labels.data(), labels.size()*sizeof(int));
	
	// Build a matrix with the observations in row:
	Mat data = asRowMatrix(images, CV_32FC1);

	

	// Number of components to keep for the PCA:
	int num_components = 100;

	// Perform a PCA:
	PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, num_components);

	// And copy the PCA results:
	Mat mean = pca.mean.clone();
	Mat eigenvalues = pca.eigenvalues.clone();
	Mat eigenvectors = pca.eigenvectors.clone();

	int k = eigenvalues.rows;
	int p = labels.size();
	
	Mat result = Mat(p, k, CV_32SC1);
	pca.project(data, result);

	// Display or save the mean photo:
	if (argc == 2) {
		cout << "argc==2" << endl;
		imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));

	}
	else {
		cout << "argc>2" << endl;
		imwrite(format("%s/mean.png", output_folder.c_str()), norm_0_255(mean.reshape(1, images[0].rows)));

	}

	// Display or save the Eigenfaces:
	for (int i = 0; i < min(num_components, eigenvectors.rows); i++) {
		//string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
		//cout << msg << endl;
		// get eigenvector #i
		Mat ev = eigenvectors.row(i).clone();
		// Reshape to original size & normalize to [0...255] for imshow.
		Mat grayscale = norm_0_255(ev.reshape(1, height));
		// Show the image & apply a Jet colormap for better sensing.
		Mat cgrayscale;
		applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
		// Display or save:
		if (argc == 2) {
			imshow(format("eigenface_%d", i), cgrayscale);

		}
		else {
			imwrite(format("%s/eigenface_%d.png", output_folder.c_str(), i), norm_0_255(cgrayscale));
		}
	}

	cout << "pca done" << endl;

	
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	// Train the SVM
	CvSVM svm;
	svm.train_auto(result, labelsMat, Mat(), Mat(), params);

	cout << "train compeleted" << endl;

	svm.save("gender_recognition(svm)");
	cout << "train saved" << endl;

	

	
	
	cin >> a;

	// the following make an eigenface recognizer
	/*

	// Display or save the image reconstruction at some predefined steps:
	for (int num_components = min(W.cols, 10); num_components < min(W.cols, 300); num_components += 15) {
		// slice the eigenvectors from the model
		Mat evs = Mat(W, Range::all(), Range(0, num_components));
		Mat projection = subspaceProject(evs, mean, images[0].reshape(1, 1));
		Mat reconstruction = subspaceReconstruct(evs, mean, projection);
		// Normalize the result:
		reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));
		// Display or save:
		if (argc == 2) {
			imshow(format("eigenface_reconstruction_%d", num_components), reconstruction);
		}
		else {
			imwrite(format("%s/eigenface_reconstruction_%d.png", output_folder.c_str(), num_components), reconstruction);
		}
	}
	
	*/


	// Display if we are not writing to an output folder:
	if (argc == 2) {
		waitKey(0);
	}


	return 0;
}