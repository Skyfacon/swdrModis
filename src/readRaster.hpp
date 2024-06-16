//
#pragma once

// #include "georaster.hpp"

#include<armadillo>
#include <gdal.h>
#include <gdal_priv.h>

inline void print_data_info(GDALDataset* poDataset)
{
	using namespace std;

	printf("Image info: \n");

	printf("Driver: %s/%s\n",
		poDataset->GetDriver()->GetDescription(),
		poDataset->GetDriver()->GetMetadataItem(GDAL_DMD_LONGNAME));

	printf("Size (row*col*band) is %dx%dx%d\n",
		poDataset->GetRasterYSize(), poDataset->GetRasterXSize(),
		poDataset->GetRasterCount());

	GDALRasterBand* pBand = poDataset->GetRasterBand(1);
	GDALDataType type = pBand->GetRasterDataType();
	cout << "Raster Type:" << type << endl;

	const char* prjInfo = poDataset->GetProjectionRef();
	if (prjInfo != nullptr)
		printf("Projection is '%s'\n", prjInfo);

	double  adfGeoTransform[6];
	if (poDataset->GetGeoTransform(adfGeoTransform) == CE_None)
	{
		printf("Origin = (%.6f,%.6f)\n", adfGeoTransform[0], adfGeoTransform[3]);
		printf("Pixel Size = (%.6f,%.6f)\n", adfGeoTransform[1], adfGeoTransform[5]);
		//cout << adfGeoTransform[2] << "  " << adfGeoTransform[4] << endl;
	}

	int  nBlockXSize, nBlockYSize;
	pBand->GetBlockSize(&nBlockXSize, &nBlockYSize);
	GDALDataType DataType = pBand->GetRasterDataType();
	GDALColorInterp color = pBand->GetColorInterpretation();

	printf("Block = %d x %d, Type = %s, ColorInterp = %s\n",
		nBlockXSize, nBlockYSize,
		GDALGetDataTypeName(DataType),
		GDALGetColorInterpretationName(color));
}


template<typename T>
int read_cube_raster(const std::string& fn, arma::Cube<T>& data, bool is_show = false)
{
	using namespace std;

	GDALAllRegister();

	const char* file = fn.c_str();
	GDALDataset  *poDataset = static_cast<GDALDataset *>(GDALOpen(file, GA_ReadOnly));
	if (poDataset == nullptr) {
		cout << "Can not read image file: " << file << endl;
		return -1;
	}

	const int nrows = poDataset->GetRasterYSize();
	const int ncols = poDataset->GetRasterXSize();
	const int nbands = poDataset->GetRasterCount();

	if (nbands < 2) {
		cout << "Error: this raster should have more than one band!\n";
		cout << "in " << file << endl;
		return -1;
	}

	const std::string c_type = typeid(T).name();
	GDALDataType dtype = convert_datatype(c_type);

	if (is_show)
	{
		cout << "Read the file: " << file << endl;
		print_data_info(poDataset);
		cout << "The used data type is " << c_type << endl;
	}


	T *pabyData = static_cast<T*>(CPLMalloc(sizeof(T)*ncols*nrows));
	if (pabyData == nullptr) {
		cout << "Cannot allocate enough memory.\n";
		return -1;
	}

	data = arma::zeros<arma::Cube<T>>(nrows, ncols, nbands);

	for (int ib = 0; ib < nbands; ++ib)
	{
		GDALRasterBand * pBand = poDataset->GetRasterBand(ib + 1);

		CPLErr ret = pBand->RasterIO(GF_Read, 0, 0, ncols, nrows,
			pabyData, ncols, nrows, dtype, 0, 0, nullptr);
		if (ret == CE_Failure) {
			cout << "Cannot read the data.\n";
			GDALClose(static_cast<GDALDatasetH>(poDataset));
			return -1;
		}

		arma::Mat<T> tmp(pabyData, ncols, nrows);
		inplace_trans(tmp);
		data.slice(ib) = tmp;
	}

	CPLFree(pabyData);
	GDALClose(static_cast<GDALDatasetH>(poDataset));

	return 0;
}


template<typename T>
void read_mat_raster(const std::string& fn, arma::Mat<T>& data, bool is_show = false)
{
	using namespace std;

	const char* file = fn.c_str();
	printf("The file name: %s\n", file);

	GDALAllRegister();
	GDALDataset  *poDataset = static_cast<GDALDataset *>(GDALOpen(file, GA_ReadOnly));
	if (poDataset == nullptr) {
		printf("Error!\n");
		exit(EXIT_FAILURE);
	}

	const std::string c_type = typeid(T).name();
	//GDALDataType dtype = convert_datatype(c_type);

	if (is_show)
	{
		cout << "Read the file: " << file << endl;
		print_data_info(poDataset);
		cout << "The used data type is " << c_type << endl;
	}

	const int nrows = poDataset->GetRasterYSize();
	const int ncols = poDataset->GetRasterXSize();

	GDALRasterBand* pBand = poDataset->GetRasterBand(1);
	GDALDataType type = pBand->GetRasterDataType();

	int basemap[1] = { 1 };
	T *pabyData = static_cast<T *>(CPLMalloc(sizeof(T)*ncols*nrows));
	poDataset->RasterIO(GF_Read, 0, 0, ncols, nrows,
		pabyData, ncols, nrows, type, 1, basemap, 0, 0, 0, nullptr);

	data = arma::Mat<T>(pabyData, ncols, nrows);
	data = data.t();
}

template<typename T>
int read_image_row(const std::string & fn, const int ir,
	arma::Mat<T>& data, bool is_show = false)
{
	using namespace std;

	GDALAllRegister();

	const char* file = fn.c_str();
	GDALDataset  *poDataset = static_cast<GDALDataset *>(GDALOpen(file, GA_ReadOnly));
	if (poDataset == nullptr) {
		printf("Can not read image file: %s!\n", file);
		return -1;
	}

	const int ncols = poDataset->GetRasterXSize();
	const int nbands = poDataset->GetRasterCount();

	if (nbands < 2) {
		printf("Error: this raster should have more than one band!\n");
		printf("in %s\n", file);
		return -1;
	}

	std::string c_type = typeid(T).name();
	GDALDataType dtype = convert_datatype(c_type);

	if (is_show)
	{
		printf("Read the file: %s\n", file);
		printf("The used data type is %s.\n", c_type.c_str());
		print_data_info(poDataset);
	}

	T *pabyData = static_cast<T*>(CPLMalloc(sizeof(T)*ncols*nbands));
	if (pabyData == nullptr) {
		printf("Cannot allocate enough memory.\n");
		return -1;
	}

	data = arma::zeros<arma::Mat<T>>(nbands, ncols);
	GDALRasterBand* pBand;

	for (int ib = 0; ib < nbands; ++ib)
	{
		pBand = poDataset->GetRasterBand(ib + 1);

		CPLErr ret = pBand->RasterIO(GF_Read, 0, ir, ncols, 1,
			pabyData, ncols, 1, dtype, 0, 0, nullptr);
		if (ret == CE_Failure) {
			printf("Cannot read the data.\n");
			GDALClose(static_cast<GDALDatasetH>(poDataset));
			return -1;
		}

		arma::Mat<T> tmp(pabyData, ncols, 1);
		inplace_trans(tmp);
		data.row(ib) = tmp;
	}

	CPLFree(pabyData);
	GDALClose(static_cast<GDALDatasetH>(poDataset));

	return 0;
}

//read 2D matrix
template<typename T>
void readMat(const std::string inFileName, T**& out_data, const long n_rows, const long n_cols)
{
	std::ifstream inFile(inFileName, std::ios::binary);
	if (!inFile)
	{
		std::cerr << "open error!" << std::endl;
		exit(1);
	}

	T *data = new T[n_cols*n_rows];
	inFile.read(reinterpret_cast<char *>(data), sizeof(T)*n_cols*n_rows);
	inFile.close();

	array2mat(data, n_rows, n_cols, out_data);

	delete[] data;
	data = nullptr;
}


template<typename T>
void array2mat(T* in_mat, long n_rows, long n_cols, T**& out_mat)
{
	if (n_rows < 1) std::cerr << "The number of row should greater 0!" << std::endl;
	if (n_cols < 1) std::cerr << "The number of column should greater 0!" << std::endl;

	for (long ir = 0; ir < n_rows; ir++)
	{
		for (long ic = 0; ic < n_cols; ic++)
		{
			out_mat[ir][ic] = in_mat[ir*n_cols + ic];
		}
	}
}

//read 3D matrix
template<typename T>
void readMat(const std::string inFileName,
	T***& outData, const int nrow, const int ncol, const int nband)
{
	using namespace std;

	ifstream inFile(inFileName, ios::binary);
	if (!inFile)
	{
		cerr << "open error!" << endl;
		exit(1);
	}

	T *data = new T[ncol*nrow*nband];
	inFile.read(reinterpret_cast<char *>(data), sizeof(T)*ncol*nrow*nband);
	inFile.close();

	array2mat(data, outData, nrow, ncol, nband);

	delete[] data;
	data = nullptr;
}


template<typename T>
void array2mat(T* in_mat, T***&out_mat, long n_rows, long n_cols, long n_bands)
{
	using namespace std;

	if (n_rows < 1) cerr << "The number of row should greater 0!" << endl;
	if (n_cols < 1) cerr << "The number of column should greater 0!" << endl;
	if (n_bands < 1) cerr << "The number of row should greater 0!" << endl;

	for (long ib = 0; ib < n_bands; ib++)
	{
		for (long ir = 0; ir < n_rows; ir++)
		{
			for (long ic = 0; ic < n_cols; ic++)
			{
				out_mat[ir][ic][ib] = in_mat[ib*n_cols*n_rows + ir * n_cols + ic];
			}
		}
	}
}


void readTIFF2FloatMat(const std::string& fn, arma::fmat& data)
{
	const char* file = fn.c_str();
	printf("The file name: %s\n", file);

	GDALAllRegister();
	GDALDataset  *poDataset = static_cast<GDALDataset *>(GDALOpen(file, GA_ReadOnly));
	if (poDataset == nullptr) {
		printf("Error!\n");
		exit(EXIT_FAILURE);
	}
	print_data_info(poDataset);

	const arma::uword nrows = poDataset->GetRasterYSize();
	const arma::uword ncols = poDataset->GetRasterXSize();
	const arma::uword nbands = poDataset->GetRasterCount();

	if (nbands != 1) {
		printf("Error: TIFF image should have only 1 band!\n");
		printf("in %s\n", file);
		exit(EXIT_FAILURE);
	}

	GDALRasterBand* pBand = poDataset->GetRasterBand(1);
	GDALDataType type = pBand->GetRasterDataType();

	if (type != GDT_Float32) {
		printf("Error: The data type of TIFF image should be float!\n");
		printf("in %s\n", file);
		exit(EXIT_FAILURE);
	}

	int basemap[1] = { 1 };
	float *pabyData = static_cast<float *>(CPLMalloc(sizeof(float)*ncols*nrows*nbands));
	poDataset->RasterIO(GF_Read, 0, 0, (int)ncols, (int)nrows, pabyData,
		(int)ncols, (int)nrows, type, (int)nbands, basemap, 0, 0, 0, nullptr);

	data = arma::fmat(pabyData, ncols, nrows, nbands);
	data = data.t();
}


bool readTIFF2FloatCube(const std::string& fn, arma::fcube& data)
{
	const char* file = fn.c_str();
	printf("The file name: %s\n", file);

	GDALAllRegister();
	GDALDataset  *poDataset = static_cast<GDALDataset *>(GDALOpen(file, GA_ReadOnly));
	if (poDataset == nullptr) {
		printf("Error!\n");
		return false;
	}
	print_data_info(poDataset);

	const arma::uword nrows = poDataset->GetRasterYSize();
	const arma::uword ncols = poDataset->GetRasterXSize();
	const arma::uword nbands = poDataset->GetRasterCount();

	if (nbands < 2) {
		printf("Error: TIFF image should have more than one band!\n");
		printf("in %s\n", file);
		return false;
	}

	GDALRasterBand* pBand = poDataset->GetRasterBand(1);
	GDALDataType type = pBand->GetRasterDataType();

	if (type != GDT_Float32) {
		printf("Error: The data type of TIFF image should be float!\n");
		printf("in %s\n", file);
		return false;
	}

	// todo
	int basemap[1] = { 1 };
	float *pabyData = static_cast<float *>(CPLMalloc(sizeof(float)*ncols*nrows*nbands));
	poDataset->RasterIO(GF_Read, 0, 0, (int)ncols, (int)nrows, pabyData,
		(int)ncols, (int)nrows, type, (int)nbands, basemap, 0, 0, 0, nullptr);

	//data = arma::fmat(pabyData, ncols, nrows, nbands);
	//data = data.t();

	return true;
}
