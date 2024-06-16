#pragma once

#include<armadillo>

#include <string>
#include <vector>


class imageGeoInfo
{
public:
	imageGeoInfo(const std::string& in_file);
	void getProjInfo(std::string& proj) const;
	void getGeoTrans(double gt[6]) const;

private:
	int readImageInfo(const std::string& imagefile);
	std::string proj_str;
	double geoTrans[6]{};
};

int glob_filelist(const std::string& in_path, std::string ext_name,
	std::vector<std::string>& filelist);
int read_3d_geotif(const std::string& filename, arma::fcube& data);
int write_3d_geotif(const arma::Cube<short>& data, const imageGeoInfo& geoinfo,
	const std::string& out_fn);