#pragma once

#include <armadillo>
#include <string>

class myConfig
{
public:
	myConfig();
	~myConfig();

	int read_config(const std::string& cfg_file);
	void print();

	std::string lut_file;
	// std::string input_file;
	// std::string out_file;

	// TOA_avg_N		number of TOA values to average with std;; default = 15
	int toa_avg_num;
	// Ref_sfactor		surface reflectance is iterated in range of [-25%, 25%]
	// of the input reflectance; default = 25 %
	float ref_range;
	// Ref_n_bins		number of reflectances in the above range, default = 5
	int ref_bin_num;
	// f_Std		std for TOA and surface cases average; times of the std; default = 1.0
	float f_std;
	int window;
	int cpu_core_num;

	arma::uvec sza_list;
	arma::uvec vza_list;
	arma::fvec dem_list;
	arma::uvec los_list;
private:
	int parse_list(const std::string& input, arma::uvec& ret) const;
	int parse_list(const std::string& input, arma::fvec& ret) const;
};