//Author: lengwanchun//
#include <iostream>


#include "ahi_swdr.h"
#include "read_config_file.h"
#include "parse_cmd_vars.h"
#include "batch_run.h"

#include <string>

int main(int argc, char* argv[])
{
	using namespace std;
	using namespace arma;

	cout << "***********************************************\n";
	cout << "\nSWDR MODIS retrieval model V1.0\n";
	cout << "\n***********************************************\n";

	//cmdVars cv;
	//if (cv.parse_cmd_vars(argc, argv) != 0) return 1;

	///for test----------------------------------------------
	//string m_cfg_file = R"(G:\29_paper1\01_validation\00_LUT\ahi_swdr_mat.cfg)";
	string m_cfg_file = R"(G:\30_paper3\01_FY3D\ahi_swdr_v3_rural_FY-3D.cfg)";
	//string m_in_path = R"(E:\21_three_poles_products\01_inputdata\test)";
	//string m_out_path = R"(E:\21_three_poles_products\02_output_result_test\test\fenkuai)";
	string m_in_path = R"(G:\30_paper3\01_FY3D\inputdata_cut)";
	string m_out_path = R"(G:\30_paper3\01_FY3D\output_cut)";
	//-------------------------------------------------------

	myConfig mycfg;
	/*mycfg.read_config(cv.m_cfg_file);*/
	mycfg.read_config(m_cfg_file);
	mycfg.print();

	//string flag;
	//cout << "\nAre arguments correct? (Y/N)" << endl;
	//cin >> flag;
	//if (flag != "Y" && flag != "y")
	//{
	//	cout << "You enter " << flag << ". ";
	//	cout << "The program exists.\n";
	//	return 1;
	//}
	//cout << "------------------------------\n";

	////const string in_path = cv.m_in_path;
	////const string out_path = cv.m_out_path;
	const string in_path = m_in_path;
	const string out_path = m_out_path;


	// // 文本形式-----------------------------------------------------
	// ahi_swdr ahisdr(mycfg);
	// string infile = R"(E:\MyWork\code\IDL_to_C\input.txt)";
	// string outfile = R"(E:\swdr.txt)";
	// ahisdr.retrieve_txt(infile, outfile);
	// return 0;
	// // -----------------------------------------------------
	
	arma::wall_clock timer;
	timer.tic();

	if (mycfg.cpu_core_num == 0)
	{
		cout << "\n => Sequential run...\n";

		ahi_swdr ahisdr(mycfg);
		if (ahisdr.sequential_run(in_path, out_path) != 0)
		{
			cout << "The failure happended in AHI_SWDR.exe.\n";
			return 1;
		}
	}
	else
	{
		cout << "\n => Parallel run...\n";
		ahi_swdr ahisdr(mycfg);
		ahisdr.batch_run_tbb(mycfg, in_path, out_path);
	}

	cout << endl;
	cout << "--------------------------------------------------\n";
	cout << "Total time: " << timer.toc() << " seconds" << endl;
	cout << "swdr_modis.exe runs successfully.\n";
	cout << "--------------------------------------------------\n";

	return 0;
}

