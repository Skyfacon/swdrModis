/*
 *
 */

#include "batch_run.h"
#include "file_io.h"
#include "ahi_swdr.h"

#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/task_scheduler_init.h"

#include<filesystem>
#include <vector>

int batch_run_tbb(const myConfig& cfg,
                  const std::string& input_path, const std::string& output_path)
{
	//using namespace std;
	//namespace fs = std::filesystem;

	//vector<string> filelist;
	//int ok = glob_filelist(input_path, ".tif", filelist);
	//if (ok != 0) return 1;

	//tbb::task_scheduler_init init(cfg.cpu_core_num);
	//const size_t file_num = filelist.size();

	//parallel_for(tbb::blocked_range<size_t>(0, file_num),
	//             [&](const tbb::blocked_range<size_t>& br)
	//             {
	//	             for (auto idx = br.begin(); idx != br.end(); idx++)
	//	             {
	//		             string in_file = filelist[idx];
	//		             cout << "-> " << in_file << endl;

	//		             fs::path mypath = in_file;
	//		             const string name = mypath.stem().u8string();

	//		             mypath = output_path;
	//		             mypath /= name + "_all_elements.tif";

	//		             if (fs::exists(mypath))
	//		             {
	//			             cout << "existing output file: " << mypath << endl;
	//			             cout << "It is being removed...\n";
	//			             fs::remove(mypath);
	//		             }

	//		             string out_file = mypath.u8string();
	//		             ahi_swdr ahisdr(cfg);
	//		             if (ahisdr.retrieve_image(in_file, out_file) != 0)
	//		             {
	//			             cerr << "cannot retrieve SWDR from file: " << in_file << endl;
	//		             }
	//	             } // end for
	//             }, tbb::auto_partitioner()); // end parallel_for

	//cout << endl;
	//cout << "--------------------------------------------------\n";
	//cout << "All data have been processed in the parallel mode.\n";
	//cout << "--------------------------------------------------\n";

	return 0;
}
