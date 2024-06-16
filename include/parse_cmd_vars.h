#pragma once

#include <string>

class cmdVars
{
public:
	cmdVars();
	int parse_cmd_vars(int argc, char * argv[]);

	std::string m_cfg_file;
	std::string m_in_path;
	std::string m_out_path;

private:
	void print();
};