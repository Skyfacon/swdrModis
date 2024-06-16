/*
 *
 */
#include "file_io.h"

#include <armadillo>
#include <gdal.h>
#include <gdal_priv.h>

#include <boost/algorithm/string.hpp>

#include <filesystem>
#include <vector>
#include <iostream>


int glob_filelist(const std::string& in_path, const std::string ext_name,
                  std::vector<std::string>& filelist)
{
	using namespace std;
	namespace fs = std::filesystem;

	const fs::path mypath(in_path);
	if (!fs::exists(mypath))
	{
		cerr << "cannot find input path: " << in_path << endl;
		return 1;
	}

	for (auto& p : fs::directory_iterator(mypath))
	{
		const fs::path& file = p.path();
		if (!fs::is_regular_file(file)) continue;

		string ext = file.extension().u8string();
		boost::algorithm::to_lower(ext);
		if (ext == ext_name)
		{
			filelist.push_back(file.u8string());
		}
		//std::cout << p.path() << '\n';
	}

	if (filelist.empty())
	{
		cerr << "No file can be found.\n";
		return 1;
	}

	return 0;
}


int read_3d_geotif(const std::string& filename, arma::fcube& data)
{
	using namespace std;
	using namespace arma;

	GDALAllRegister();

	const char* file = filename.c_str();
	GDALDataset* poDataset = static_cast<GDALDataset *>(GDALOpen(file, GA_ReadOnly));
	if (poDataset == nullptr)
	{
		cout << "Can not read image file: " << file << endl;
		return 1;
	}

	const int nrows = poDataset->GetRasterYSize();
	const int ncols = poDataset->GetRasterXSize();
	const int nbands = poDataset->GetRasterCount();

	if (nbands < 2)
	{
		cout << "Error: this raster should have more than one band!\n";
		cout << "in " << file << endl;
		return -1;
	}

	float* pabyData = static_cast<float*>(CPLMalloc(sizeof(float) * ncols * nrows));
	if (pabyData == nullptr)
	{
		cout << "Cannot allocate enough memory.\n";
		return 1;
	}

	data = arma::zeros<arma::fcube>(nrows, ncols, nbands);

	for (int ib = 0; ib < nbands; ++ib)
	{
		GDALRasterBand* pBand = poDataset->GetRasterBand(ib + 1);

		CPLErr ret = pBand->RasterIO(GF_Read, 0, 0, ncols, nrows,
		                             pabyData, ncols, nrows, GDT_Float32, 0, 0, nullptr); //float:GDT_Float32  double:GDT_Float64
		if (ret == CE_Failure)
		{
			cout << "Cannot read the data.\n";
			GDALClose(static_cast<GDALDatasetH>(poDataset));
			return 1;
		}

		arma::fmat tmp(pabyData, ncols, nrows);
		inplace_trans(tmp);
		data.slice(ib) = tmp;
	}

	CPLFree(pabyData);
	GDALClose(static_cast<GDALDatasetH>(poDataset));

	return 0;
}


int write_3d_geotif(const arma::Cube<short>& data, const imageGeoInfo& geoinfo,
                    const std::string& out_fn)
{
	const short fillvalue = -1;

	GDALAllRegister();

	
	double geoTrans[6];
	geoinfo.getGeoTrans(geoTrans);

	std::string proj;
	geoinfo.getProjInfo(proj);

	const char* pszFormat = "GTiff";
	GDALDriver* poDriver = GetGDALDriverManager()->GetDriverByName(pszFormat);
	if (poDriver == nullptr)
	{
		printf("[Error] GDAL does't support data format %s. \n", pszFormat);
		return 1;
	}

	char** papszOptions = nullptr;
	papszOptions = CSLSetNameValue(papszOptions, "INTERLEAVE", "BAND");
	papszOptions = CSLSetNameValue(papszOptions, "COMPRESS", "LZW");

	const int nrows = data.n_rows;
	const int ncols = data.n_cols;
	const int nbnds = data.n_slices;

	GDALDataset* poDataset = poDriver->Create(out_fn.c_str(),
	                                          ncols, nrows, nbnds,
	                                          GDT_Int16, papszOptions);
	if (poDataset == nullptr)
	{
		printf("[Error] GDAL dataset cannot be created.\n");
		return 1;
	}

	if (proj.length() > 0)
	{
		poDataset->SetProjection(proj.c_str());
		poDataset->SetGeoTransform(geoTrans);
	}

	for (int ib = 0; ib < nbnds; ++ib)
	{
		arma::Mat<short> band_data = data.slice(ib);
		arma::inplace_trans(band_data);

		short* pabyData = band_data.memptr();

		GDALRasterBand* pBand = poDataset->GetRasterBand(ib + 1);
		CPLErr ret = pBand->RasterIO(GF_Write, 0, 0, ncols, nrows,
		                             pabyData, ncols, nrows, GDT_Int16, 0, 0, nullptr);
		if (ret == CE_Failure)
		{
			printf("Cannot read the data.\n");
			GDALClose(static_cast<GDALDatasetH>(poDataset));
			return 1;
		}
		pBand->SetNoDataValue(fillvalue);
	} // endfor ib

	GDALClose(static_cast<GDALDatasetH>(poDataset));

	return 0;
}

imageGeoInfo::imageGeoInfo(const std::string& in_file)
{
	int ok = readImageInfo(in_file);
	if (ok != 0) exit(EXIT_FAILURE);
}

int imageGeoInfo::readImageInfo(const std::string& imagefile)
{
	GDALAllRegister();

	const char* file = imagefile.c_str();
	auto* poDataset = static_cast<GDALDataset *>(GDALOpen(file, GA_ReadOnly));
	if (poDataset == nullptr)
	{
		std::cout << "Cannot read image file: " << file << std::endl;
		return 1;
	}

	const char* prjInfo = poDataset->GetProjectionRef();
	if (prjInfo == nullptr)
	{
		std::cout << "The image file has no projection.\n";
		proj_str = "";
	}
	else
	{
		proj_str = std::string(prjInfo);
	}

	double adfGeoTransform[6];
	const CPLErr ok = poDataset->GetGeoTransform(adfGeoTransform);
	if (ok == CE_None) // successful
	{
		for (int i = 0; i < 6; ++i)
			geoTrans[i] = adfGeoTransform[i];
	}
	else
	{
		for (double& geoTran : geoTrans)
			geoTran = 0.0;
	}

	return 0;
}

inline void imageGeoInfo::getProjInfo(std::string& proj) const
{
	proj = proj_str;
}

inline void imageGeoInfo::getGeoTrans(double gt[6]) const
{
	for (int i = 0; i < 6; ++i)
	{
		gt[i] = geoTrans[i];
	}
}
