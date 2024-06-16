#pragma once

#include "georaster.hpp"

#include <armadillo>
#include "gdal.h"
#include "gdal_priv.h"

#include <string>

namespace gsio
{
	template<typename T>
	int write_cube_GTiff(const georaster<T> & geoimg, const std::string& out_fn,
		T fillvalue = 0, bool is_show = false)
	{
		GDALAllRegister();

		auto data = geoimg.m_cube;

		double geoTrans[6];
		geoimg.getGeoTrans(geoTrans);

		std::string proj;
		geoimg.getProjInfo(proj);

		const char *pszFormat = "GTiff";
		GDALDriver *poDriver = GetGDALDriverManager()->GetDriverByName(pszFormat);
		if (poDriver == nullptr) {
			printf("[Error] GDAL does't support data format %s. \n", pszFormat);
			return 1;
		}

		char** papszMetadata = poDriver->GetMetadata();
		if (is_show)
		{
			if (CSLFetchBoolean(papszMetadata, GDAL_DCAP_CREATE, FALSE))
				printf("Driver %s supports Create() method.\n", pszFormat);
			if (CSLFetchBoolean(papszMetadata, GDAL_DCAP_CREATECOPY, FALSE))
				printf("Driver %s supports CreateCopy() method.\n", pszFormat);
		}

		char **papszOptions = nullptr;
		papszOptions = CSLSetNameValue(papszOptions, "INTERLEAVE", "BAND");
		papszOptions = CSLSetNameValue(papszOptions, "COMPRESS", "LZW");

		const int nrows = data.n_rows;
		const int ncols = data.n_cols;
		const int nbnds = data.n_slices;

		std::string c_type = typeid(T).name();
		GDALDataType dtype = gsio::convert_datatype(c_type);

		if (is_show)
			printf("The data type is %s.\n", c_type.c_str());

		GDALDataset* poDataset = poDriver->Create(out_fn.c_str(), ncols, nrows, nbnds, dtype, papszOptions);
		if (poDataset == nullptr) {
			printf("[Error] GDAL dataset cannot be created.\n");
			return 1;
		}

		if (proj.length() > 0)
		{
			poDataset->SetProjection(proj.c_str());
			poDataset->SetGeoTransform(geoTrans);
		}

		GDALRasterBand* pBand;
		arma::Mat<T> tmp;
		for (int ib = 0; ib < nbnds; ++ib)
		{
			tmp = data.slice(ib);
			arma::inplace_trans(tmp);

			T *pabyData = tmp.memptr();

			pBand = poDataset->GetRasterBand(ib + 1);
			CPLErr ret = pBand->RasterIO(GF_Write, 0, 0, ncols, nrows,
				pabyData, ncols, nrows, dtype, 0, 0, nullptr);
			if (ret == CE_Failure) {
				printf("Cannot read the data.\n");
				GDALClose(static_cast<GDALDatasetH>(poDataset));
				return 1;
			}
			pBand->SetNoDataValue(fillvalue);
		}

		GDALClose(static_cast<GDALDatasetH>(poDataset));

		return 0;
	}


	template<typename T>
	int write_mat_GTiff(const gsio::georaster<T> & geoimg, const std::string& out_fn,
		const T fillvalue = 0, bool is_show = false)
	{
		GDALAllRegister();

		auto data = geoimg.m_mat;
		//arma::Mat<T> data;
		//geoimg.getMat(data);

		double geoTrans[6];
		geoimg.getGeoTrans(geoTrans);

		std::string proj;
		geoimg.getProjInfo(proj);

		const char *pszFormat = "GTiff";
		GDALDriver *poDriver = GetGDALDriverManager()->GetDriverByName(pszFormat);
		if (poDriver == nullptr) {
			printf("[Error] GDAL does't support data format %s. \n", pszFormat);
			return 1;
		}

		char** papszMetadata = poDriver->GetMetadata();
		if (is_show)
		{
			if (CSLFetchBoolean(papszMetadata, GDAL_DCAP_CREATE, FALSE))
				printf("Driver %s supports Create() method.\n", pszFormat);
			if (CSLFetchBoolean(papszMetadata, GDAL_DCAP_CREATECOPY, FALSE))
				printf("Driver %s supports CreateCopy() method.\n", pszFormat);
		}

		char **papszOptions = nullptr;
		papszOptions = CSLSetNameValue(papszOptions, "INTERLEAVE", "BAND");
		papszOptions = CSLSetNameValue(papszOptions, "COMPRESS", "LZW");

		const int nrows = data.n_rows;
		const int ncols = data.n_cols;

		std::string c_type = typeid(T).name();
		GDALDataType dtype = gsio::convert_datatype(c_type);

		if (is_show)
			printf("The data type is %s.\n", c_type.c_str());

		GDALDataset* poDataset = poDriver->Create(out_fn.c_str(), ncols, nrows, 1, dtype, papszOptions);
		if (poDataset == nullptr) {
			printf("[Error] GDAL dataset cannot be created.\n");
			return 1;
		}

		if (proj.length() > 0)
		{
			poDataset->SetProjection(proj.c_str());
			poDataset->SetGeoTransform(geoTrans);
		}

		arma::inplace_trans(data);
		T *pabyData = data.memptr();

		GDALRasterBand * pBand = poDataset->GetRasterBand(1);
		CPLErr ret = pBand->RasterIO(GF_Write, 0, 0, ncols, nrows,
			pabyData, ncols, nrows, dtype, 0, 0, nullptr);
		if (ret == CE_Failure) {
			printf("Cannot read the data.\n");
			GDALClose(static_cast<GDALDatasetH>(poDataset));
			return 1;
		}
		pBand->SetNoDataValue(fillvalue);

		GDALClose(static_cast<GDALDatasetH>(poDataset));

		return 0;
	}

	//write the 3D mat in envi binary format
	template<typename T>
	int save_envi_binary(const std::string file_name,
		T*** data, const int nrow, const int ncol, const int nband)
	{
		FILE * pWrite = fopen(file_name.c_str(), "wb");
		if (pWrite == nullptr)
		{
			fclose(pWrite);
			return 1;
		}
		for (int ib = 0; ib < nband; ib++)
		{
			for (int ir = 0; ir < nrow; ir++)
			{
				for (int ic = 0; ic < ncol; ic++)
				{
					fwrite(&data[ir][ic][ib], sizeof(float), 1, pWrite);
				}
			}
		}
		fclose(pWrite);
		return 0;
	}

	//write the 2D mat in envi binary format
	template<typename T>
	int save_envi_binary(const std::string file_name, T** data, const int nrow, const int ncol)
	{
		using namespace std;

		ofstream outFile(file_name, ios::binary);
		if (!outFile.good())
		{
			cout << "can not open this file!\n" << file_name << endl;
			return -1;
		}

		for (int ir = 0; ir < nrow; ir++)
		{
			for (int ic = 0; ic < ncol; ic++)
			{
				outFile.write((char*)&data[ir][ic], sizeof(T));
			}
		}
		outFile.close();
		return 0;
	}

	//write the 2D mat in ASCII format
	template<typename T>
	int save_ascii(const std::string file_name, T** data, const int nrow, const int ncol)
	{
		using namespace std;

		ofstream out_fn;
		out_fn.open(file_name);

		if (out_fn.is_open())
		{
			for (int ir = 0; ir < nrow; ir++)
			{
				for (int ic = 0; ic < ncol; ic++)
				{
					out_fn << data[ir][ic] << endl;
				}
			}
		}
		else
		{
			cout << "can not open this file\n" << file_name << endl;
			out_fn.close();
			return -1;
		}

		out_fn.close();
		return 0;
	}
} // end of namespace gsio


