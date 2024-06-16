#include "read_config_file.h"
#include <boost/algorithm/string.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>

myConfig::myConfig()
{
	// TOA_avg_N		number of TOA values to average with std;; default = 15
	int toa_avg_num = 15;
	// Ref_sfactor		surface reflectance is iterated in range of [-25%, 25%]
	// of the input reflectance; default = 25 %
	float ref_range = 0.25;
	// Ref_n_bins		number of reflectances in the above range, default = 5
	int ref_bin_num = 5;
	// f_Std		std for TOA and surface cases average; times of the std; default = 1.0
	float f_std = 1.0;
}

myConfig::~myConfig() = default;

int myConfig::read_config(const std::string& cfg_file)
{
	using namespace std;
	namespace fs = std::filesystem;

	if (cfg_file.length() == 0)
	{
		cout << "[Error] config file is empty.\n";
		return 1;
	}

	const fs::path mypath{cfg_file};
	if (! fs::exists(mypath))
	{
		cout << "[Error] cannot find config file: " << cfg_file << endl;
		return 1;
	}

	//cout << "Read config file: " << cfg_file << endl;

	ifstream fid{cfg_file};
	if (fid.is_open() != 1)
	{
		cout << "Cannot open input yaml file: " << cfg_file << endl;
		return 1;
	}

	string tmp;
	vector<string> data;
	while (!fid.eof())
	{
		getline(fid, tmp);
		data.push_back(tmp);
	}
	fid.close();

	const size_t nrows = data.size();
	for (size_t ir = 0; ir < nrows; ir++)
	{
		string line = data[ir];

		boost::trim(line);
		if (line.length() == 0) continue;

		const string head = line.substr(0, 1);
		if (head == "#") continue;
		// cout << "> " << line << endl;

		vector<string> pack;
		boost::split(pack, line, boost::is_any_of("="), boost::token_compress_on);
		if (pack.size() != 2)
		{
			cout << "[Error] parse wrong: \n";
			for (const auto& item : pack)
				cout << item << " ";
			cout << endl;
		}

		string key = pack[0];
		string value = pack[1];
		boost::trim(key);
		boost::trim(value);
		boost::algorithm::to_lower(key);
		// cout <<  key << " -> " << value << endl;

		if (key == "lut_file")
			lut_file = value;
		else if (key == "toa_avg_num")
			toa_avg_num = stoi(value);
		else if (key == "ref_range")
			ref_range = stof(value);
		else if (key == "ref_bin_num")
			ref_bin_num = stoi(value);
		else if (key == "f_std")
			f_std = stof(value);
		else if (key == "window")
		{
			window = stoi(value);
			if (window < 0)
			{
				cout << "The argument 'window' is " << window << endl;
				cerr << "The argument 'window' cannot be negative.";
				return 1;
			}
		}
		else if (key == "cpu_core_num")
		{
			cpu_core_num = stoi(value);
			if (window < 0)
			{
				cout << "The argument 'cpu_core_num' is " << cpu_core_num << endl;
				cerr << "The argument 'cpu_core_num' cannot be negative.";
				return 1;
			}
		}
		else if (key == "sza_list")
		{
			if (parse_list(value, sza_list) != 0) return 1;
		}
		else if (key == "vza_list")
		{
			if (parse_list(value, vza_list) != 0) return 1;
		}
		else if (key == "dem_list")
		{
			if (parse_list(value, dem_list) != 0) return 1;
		}
		else if (key == "los_list")
		{
			if (parse_list(value, los_list) != 0) return 1;
		}
		else
		{
			cout << "[Error] cannot find the correct key: " << key
				<< " -> " << value << endl;
			return 1;
		}
	}

	return 0;
}

void myConfig::print()
{
	using namespace std;

	cout << "\nParse config file arguments: \n";
	cout << "------------------------------\n";
	cout << "LUT file     : " << lut_file << endl;
	cout << "toa_avg_num  : " << toa_avg_num << endl;
	cout << "ref_range    : " << ref_range << endl;
	cout << "ref_bin_num  : " << ref_bin_num << endl;
	cout << "f_std        : " << f_std << endl;

	cout << "SZA list     : ";
	for (const auto& item : sza_list)
		cout << item << ", ";
	cout << endl;

	cout << "VZA list     : ";
	for (const auto& item : vza_list)
		cout << item << ", ";
	cout << endl;

	cout << "DEM list     : ";
	for (const auto& item : dem_list)
		cout << item << ", ";
	cout << endl;

	cout << "LOS list     : ";
	for (const auto& item : los_list)
		cout << item << ", ";
	cout << endl;

	if (window == 0)
		printf("window       : %d    ==> no smooth.\n", window);
	else
		printf("window       : %d    ==> %d * %d smooth.\n",
		       window, (window * 2) + 1, (window * 2) + 1);

	if (cpu_core_num == 0)
		printf("CPU number   : %d    ==> sequential run.\n", cpu_core_num);
	else
		printf("CPU number   : %d    ==> parallel run.\n", cpu_core_num);

	cout << "------------------------------\n";
}

int myConfig::parse_list(const std::string& input, arma::uvec& ret) const
{
	using namespace std;

	vector<string> pack;
	boost::split(pack, input, boost::is_any_of(","), boost::token_compress_on);

	ret = arma::zeros<arma::uvec>(pack.size());
	for (size_t idx = 0; idx < pack.size(); idx++)
	{
		string item = pack[idx];
		boost::trim(item);
		if (item.length() == 0)
		{
			cout << "[Error] item is empty for " << input << endl;
			return 1;
		}

		ret(idx) = stoi(item);
	}

	return 0;
}

int myConfig::parse_list(const std::string& input, arma::fvec& ret) const
{
	using namespace std;

	vector<string> pack;
	boost::split(pack, input, boost::is_any_of(","), boost::token_compress_on);

	ret = arma::zeros<arma::fvec>(pack.size());
	for (size_t idx = 0; idx < pack.size(); idx ++)
	{
		string item = pack[idx];
		boost::trim(item);
		if (item.length() == 0)
		{
			cout << "[Error] item is empty for " << input << endl;
			return 1;
		}

		ret(idx) = stof(item);
	}

	return 0;
}
