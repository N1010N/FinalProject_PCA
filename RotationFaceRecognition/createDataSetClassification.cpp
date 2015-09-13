#include<windows.h>
#include<regex>
#include<stdio.h>
#include<string>
#include<fstream>
#include<iostream>
#include <sstream>

using namespace std;

string sourceAddress = "D:\\dataset\\face\\original";
string desAddress = "F:\\nastaran\\university\\Proje payani\\RotationFaceRecognition\\Resources\\Dataset1";
int numClassification = 200;
int num_rotation = 14;
ofstream out(desAddress + "ImageInformation.txt", ios::out);

int main()
{
	WIN32_FIND_DATA fd;
	string pathImages = sourceAddress+"\\";
	string str = sourceAddress + "\\*.jpg";

	char currentPath[500];
	strcpy_s(currentPath, str.c_str());
	HANDLE hFind = ::FindFirstFile(currentPath, &fd);
	if (hFind != INVALID_HANDLE_VALUE)
		for (int i = 1; i <=numClassification; i++)
	{
			for (int j = 1; j <= num_rotation;j++)
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {

				//std::regex_search(s.begin(), s.end(), match, rgx)
					out << pathImages<<fd.cFileName << ";" << j << endl;
					::FindNextFile(hFind, &fd);
				//names.push_back(fd.cFileName);
			}
	}
	::FindClose(hFind);
	out.close();
	return 0;
}
