#include "parse_cmd_vars.h"

#include <boost/algorithm/string.hpp>

#include <filesystem>
#include <iostream>
#include <string>


cmdVars::cmdVars(): m_cfg_file(""),m_in_path(""),m_out_path("")
{
}

int cmdVars::parse_cmd_vars(int argc, char * argv[])
{
	using namespace std;
	namespace fs = std::filesystem;

	if (argc != 4)
	{
		cout << "The input argument is wrong.\n";
		cout << "ahi_swdr.exe cfg=[xxx].cfg in_path=xxx out_path=xxx\n";
		return 1;
	}

	for (int idx = 1; idx < argc; idx++)
	{
		string var = argv[idx];
		boost::trim(var);
		//cout << var << endl;

		vector<string> couple;
		boost::split(couple, var, boost::is_any_of("="), boost::token_compress_on);
		if (couple.size() != 2)
		{
			cerr << "Wrong cmd parameter setting: " << var << endl;
			return 1;
		}

		string var_key = couple[0];
		boost::to_lower(var_key);
		string var_value = couple[1];
		// cout << key << " => " << value;
		fs::path mypath;

		if (var_key == "cfg")
		{
			if (var_value.length() < 3)
			{
				cerr << "cfg file is wrong: " << var_value << endl;
				return 1;
			}

			mypath = var_value;
			if (! fs::exists(mypath))
			{
				cerr << "cannot find cfg file: " << var_value << endl;
				return 1;
			}

			m_cfg_file = var_value;
		}
		else if( var_key == "ip")
		{
			if (var_value.length() == 0)
			{
				cerr << "input path is null." << endl;
				return 1;
			}

			mypath = var_value;
			if (!fs::exists(mypath))
			{
				cerr << "cannot find input path: " << var_value << endl;
				return 1;
			}

			m_in_path= var_value;
		}
		else if (var_key == "op")
		{
			if (var_value.length() == 0)
			{
				cerr << "output path is null." << endl;
				return 1;
			}

			mypath = var_value;
			if (!fs::exists(mypath))
			{
				cerr << "cannot find output path: " << var_value << endl;
				return 1;
			}

			m_out_path = var_value;
		}
		else
		{
			cerr << "cmd argument is wrong." << endl;
			cerr << var_key << " : " << var_value << endl;
			return 1;
		}
	} // endfor

	print();

	return 0;
}

void cmdVars::print()
{
	using namespace std;

	cout << "\n-----------------------------\n";
	cout << "CMD parameters: \n";
	cout << "The cfg file:    " << m_cfg_file << endl;
	cout << "The input path:  " << m_in_path << endl;
	cout << "The output path: " << m_out_path << endl;
	cout << "\n-----------------------------\n";
}
