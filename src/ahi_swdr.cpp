//
#include "file_io.h"
#include "ahi_swdr.h"

#include <boost/algorithm/string.hpp>
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/task_scheduler_init.h"

#include <fstream>
#include <iostream>
#include <filesystem>
#include <vector>



ahi_swdr::ahi_swdr(const myConfig& cfg) :
	m_lut_file(cfg.lut_file), m_toa_avg_num(cfg.toa_avg_num),
	m_ref_range(cfg.ref_range), m_ref_bin_num(cfg.ref_bin_num),
	m_f_std(cfg.f_std), m_window(cfg.window),
	m_sza_list(cfg.sza_list), m_vza_list(cfg.vza_list),
	m_dem_list(cfg.dem_list), m_los_list(cfg.los_list),
	m_sza_list_ft(arma::conv_to<arma::fvec>::from(m_sza_list)),
	m_vza_list_ft(arma::conv_to<arma::fvec>::from(m_vza_list)),
	m_los_list_ft(arma::conv_to<arma::fvec>::from(m_los_list))
{
	m_up_sza = 0;
	m_dw_sza = 0;
	m_up_vza = 0;
	m_dw_vza = 0;
	m_up_dem = 0;
	m_dw_dem = 0;
	m_up_dem10 = 0;
	m_dw_dem10 = 0;
	m_min_los = 0;

	//���˲��ұ��Ĳ���
	//idx_filter_sza = 87480;
	//idx_filter_vza = 9720;
	//idx_filter_los = 1944;
	//idx_filter_dem = 324;

	idx_filter_sza = 86400;
	idx_filter_vza = 10800;
	idx_filter_los = 2160;
	idx_filter_dem = 360;

	//���ұ�����
	lut_cols = 40; 


	std::cout << "Begin to read LUT file: " << m_lut_file << std::endl;
	if (read_lut(m_lut_file, lut_cols) != 0) exit(EXIT_FAILURE);
	std::cout << "LUT file have been read.\n";

	m_sza_min = m_sza_list(0);
	m_sza_max = m_sza_list(m_sza_list.n_elem - 1);

	m_vza_min = m_vza_list(0);
	m_vza_max = m_vza_list(m_vza_list.n_elem - 1);

	m_min_dem = m_dem_list(0);
	m_max_dem = m_dem_list(m_dem_list.n_elem - 1);

}

ahi_swdr::~ahi_swdr() = default;

int ahi_swdr::sequential_run(const std::string& input_path, const std::string& output_path)
{
	using namespace std;
	namespace fs = std::filesystem;

	vector<string> filelist;
	int ok = glob_filelist(input_path, ".tif", filelist);
	if (ok != 0) return 1;

	for (auto& in_file : filelist)
	{
		cout << "-> " << in_file << endl;

		fs::path mypath = in_file;
		const string name = mypath.stem().u8string();

		mypath = output_path;
		mypath /= name + "_all_elements.tif";

		if (fs::exists(mypath))
		{
			cout << "existing output file: " << mypath << endl;
			cout << "It is being removed...\n";
			fs::remove(mypath);
		}

		string out_file = mypath.u8string();
		if (retrieve_image(in_file, out_file) != 0)
		{
			cerr << "cannot retrieve SWDR from file: " << in_file << endl;
		}
		//return 0; //only_single_image
	} // for

	cout << endl;
	cout << "--------------------------------------------------\n";
	cout << "All data have been processed in the sequential mode.\n";
	cout << "--------------------------------------------------\n";

	return 0;
}

int ahi_swdr::batch_run_tbb(const myConfig& cfg,
	const std::string& input_path, const std::string& output_path)
{
	using namespace std;
	namespace fs = std::filesystem;

	vector<string> filelist;
	int ok = glob_filelist(input_path, ".tif", filelist);
	if (ok != 0) return 1;

	tbb::task_scheduler_init init(cfg.cpu_core_num);
	const size_t file_num = filelist.size();

	parallel_for(tbb::blocked_range<size_t>(0, file_num),
		[&](const tbb::blocked_range<size_t>& br)
	{
		for (auto idx = br.begin(); idx != br.end(); idx++)
		{
			string in_file = filelist[idx];
			cout << "-> " << in_file << endl;

			fs::path mypath = in_file;
			const string name = mypath.stem().u8string();

			mypath = output_path;
			mypath /= name + "_all_elements.tif";

			if (fs::exists(mypath))
			{
				cout << "existing output file: " << mypath << endl;
				cout << "It is being removed...\n";
				fs::remove(mypath);
			}

			string out_file = mypath.u8string();
			ahi_swdr ahisdr(cfg);
			if (ahisdr.retrieve_image(in_file, out_file) != 0)
			//if (retrieve_image(in_file, out_file) != 0)
			{
				cerr << "cannot retrieve SWDR from file: " << in_file << endl;
			}
		} // end for
	}, tbb::auto_partitioner()); // end parallel_for

	cout << endl;
	cout << "--------------------------------------------------\n";
	cout << "All data have been processed in the parallel mode.\n";
	cout << "--------------------------------------------------\n";

	return 0;
}


int ahi_swdr::retrieve_image(const std::string& input_file, const std::string& out_file)
{
	using namespace std;
	using namespace arma;
	namespace fs = std::filesystem;

	fcube data;
	int ok = read_3d_geotif(input_file, data);
	if (ok != 0) return 1;
	cout << input_file << " have been read.\n";


	//hdf
	const fmat flag_mat = data.slice(0);
	const fmat sza_mat = data.slice(1) / 100.0; // in degree
	const fmat vza_mat = data.slice(2) / 100.0; // in degree
	const fmat los_mat = data.slice(3) / 100.0; // in degree
	const fmat dem_mat = data.slice(4) / 1000.0; // in km

	//const fmat toa_rad_mat_b1 = data.slice(5) / 10.0;
	//const fmat toa_rad_mat_b2 = data.slice(6) / 10.0;
	//const fmat toa_rad_mat_b3 = data.slice(7) / 10.0; //toa_rad_mat
	//const fmat toa_rad_mat_b4 = data.slice(8) / 10.0;
	const fmat toa_rad_mat_b1 = data.slice(7) / 10.0;
	const fmat toa_rad_mat_b2 = data.slice(8) / 10.0;
	const fmat toa_rad_mat_b3 = data.slice(5) / 10.0; //toa_rad_mat
	const fmat toa_rad_mat_b4 = data.slice(6) / 10.0;
	const fmat toa_rad_mat_b5 = data.slice(9) / 100.0;
	const fmat toa_rad_mat_b6 = data.slice(10) / 100.0;
	const fmat toa_rad_mat_b7 = data.slice(11) / 500.0;

	const fmat band1_ref_mat = data.slice(12) / 1000.0;
	const fmat band2_ref_mat = data.slice(13) / 1000.0;
	const fmat band3_ref_mat = data.slice(14) / 1000.0; //blue_ref_mat
	const fmat band4_ref_mat = data.slice(15) / 1000.0;
	const fmat band5_ref_mat = data.slice(16) / 1000.0;
	const fmat band6_ref_mat = data.slice(17) / 1000.0;
	const fmat band7_ref_mat = data.slice(18) / 1000.0;

	const fmat sw_alb_mat = data.slice(19) / 1000.0;
	const fmat vis_alb_mat = data.slice(20) / 1000.0; //
	
	const uword nrows = flag_mat.n_rows;
	const uword ncols = flag_mat.n_cols;

	// -1 for invalid data
	fmat swdr_mat = zeros<fmat>(nrows, ncols) - 1.0;
	fmat sw_dir_mat = zeros<fmat>(nrows, ncols) - 1.0;
	fmat par_mat = zeros<fmat>(nrows, ncols) - 1.0;
	fmat pardir_mat = zeros<fmat>(nrows, ncols) - 1.0;
	fmat uva_mat = zeros<fmat>(nrows, ncols) - 1.0;
	fmat uvb_mat = zeros<fmat>(nrows, ncols) - 1.0;
	fmat toa_up_flux_mat = zeros<fmat>(nrows, ncols) - 1.0;
	fmat rho_mat = zeros<fmat>(nrows, ncols) - 1.0;

	//�������,�ȶ�άתһά//��һάת��ά
	fvec sza_mat_v = sza_mat.as_col();
	fvec myvza_mat_v = vza_mat.as_col();
	//��vec����һ�����������У��
	myvza_mat_v = arma::sin(arma::datum::pi * myvza_mat_v / 180.0);
	myvza_mat_v = myvza_mat_v * 6371.0 / 6471.0;
	fvec vza_mat_v = 180.0 * arma::asin(myvza_mat_v) / arma::datum::pi;

	fvec los_mat_v = los_mat.as_col();
	fvec dem_mat_v = dem_mat.as_col();
	fvec flag_mat_v = flag_mat.as_col();

	fvec toa_rad_mat_b1_v = toa_rad_mat_b1.as_col();
	fvec toa_rad_mat_b2_v = toa_rad_mat_b2.as_col();
	fvec toa_rad_mat_b3_v = toa_rad_mat_b3.as_col();
	fvec toa_rad_mat_b4_v = toa_rad_mat_b4.as_col();
	fvec toa_rad_mat_b5_v = toa_rad_mat_b5.as_col();
	fvec toa_rad_mat_b6_v = toa_rad_mat_b6.as_col();
	fvec toa_rad_mat_b7_v = toa_rad_mat_b7.as_col();

	fvec band1_ref_mat_v = band1_ref_mat.as_col();
	fvec band2_ref_mat_v = band2_ref_mat.as_col();
	fvec band3_ref_mat_v = band3_ref_mat.as_col();
	fvec band4_ref_mat_v = band4_ref_mat.as_col();
	fvec band5_ref_mat_v = band5_ref_mat.as_col();
	fvec band6_ref_mat_v = band6_ref_mat.as_col();
	fvec band7_ref_mat_v = band7_ref_mat.as_col();
	fvec sw_alb_mat_v = sw_alb_mat.as_col();
	fvec vis_alb_mat_v = vis_alb_mat.as_col();

	fvec swdr_v = swdr_mat.as_col();
	fvec sw_dir_v = sw_dir_mat.as_col();
	fvec par_v = par_mat.as_col();
	fvec pardir_v = pardir_mat.as_col();
	fvec uva_v = uva_mat.as_col();
	fvec uvb_v = uvb_mat.as_col();
	fvec toa_up_flux_v = toa_up_flux_mat.as_col();
	fvec rho_v = rho_mat.as_col();

	//�ҵ�Ӱ���Ӧ��sza���ֵ��Сֵ
	float sza_min_image = sza_mat_v.min();
	float sza_max_image = sza_mat_v.max();

	uword up_sza_idx_image;
	uword dw_sza_idx_image;
	int flag = filter_sza(sza_min_image, sza_max_image, up_sza_idx_image, dw_sza_idx_image, m_sza_list_ft, m_sza_min, m_sza_max);
	if (flag != 0) return 1;

	//�ҵ�Ӱ���Ӧ��vza���ֵ��Сֵ
	float vza_min_image = vza_mat_v.min();
	float vza_max_image = vza_mat_v.max();

	uword up_vza_idx_image;
	uword dw_vza_idx_image;
	flag = filter_sza(vza_min_image, vza_max_image, up_vza_idx_image, dw_vza_idx_image, m_vza_list_ft, m_vza_min, m_vza_max);
	if (flag != 0) return 1;

	//---------------------------------------------------------

	cout << "Begin to retrieve SWDR from MODIS data...\n";
	wall_clock timer;

	//==�������===========
	timer.tic();
	//----------------------------------------
	for (uword i = dw_sza_idx_image; i < up_sza_idx_image; i++) //10��,���ֵ��85
	{
		uword up_sza_idx;
		uword dw_sza_idx;
		uword up_vza_idx;
		uword dw_vza_idx;
		uword up_dem_idx;
		uword dw_dem_idx;
		uword los_idx;

		//------------------------------------------------
		dw_sza_idx = i;
		up_sza_idx = i + 1;
		m_dw_sza = m_sza_list(i);
		m_up_sza = m_sza_list(i+1);

		//dw_SZA_indx
		uword st = dw_sza_idx * idx_filter_sza;
		uword ed = st + idx_filter_sza - 1;
		fmat lut_ds = m_lut.rows(span(st, ed)); //lut1
		//up_SZA_indx
		st = up_sza_idx * idx_filter_sza;
		ed = st + idx_filter_sza - 1;
		fmat lut_us = m_lut.rows(span(st, ed)); //lut3

		//==================================================
		//VZA
		for (uword j = dw_vza_idx_image; j < up_vza_idx_image; j++)
		{
			dw_vza_idx = j;
			up_vza_idx = j + 1;
			m_dw_vza = m_vza_list(j);
			m_up_vza = m_vza_list(j + 1);

			//-----------------------------
			uvec idx_tile = find(sza_mat_v <= m_up_sza && sza_mat_v >= m_dw_sza && vza_mat_v <= m_up_vza && vza_mat_v >= m_dw_vza && flag_mat_v == 1);
			if (idx_tile.n_elem < 5) continue;
			//-----------------------------
			//
			//(1)dw_SZA_dw_VZA_indx
			st = dw_vza_idx * idx_filter_vza;
			ed = st + idx_filter_vza - 1;
			fmat lut_ds_dv = lut_ds.rows(span(st, ed));
			//(2)dw_SZA_up_VZA_indx
			st = up_vza_idx * idx_filter_vza;
			ed = st + idx_filter_vza - 1;
			fmat lut_ds_uv = lut_ds.rows(span(st, ed));
			//(3)up_SZA_dw_VZA_indx
			st = dw_vza_idx * idx_filter_vza;
			ed = st + idx_filter_vza - 1;
			fmat lut_us_dv = lut_us.rows(span(st, ed));
			//(4)up_SZA_up_vza_idx
			st = up_vza_idx * idx_filter_vza;
			ed = st + idx_filter_vza - 1;
			fmat lut_us_uv = lut_us.rows(span(st, ed));

			//==================================================
			//LOS
			for (uword m = 0; m < 5; m++)
			{
				float m_dw_los = 45 * m - 22.5;
				float m_up_los = 45 * m + 22.5;
				if (m_dw_los < 0) // vza < min
				{
					m_dw_los = 0;
				}
				if (m_up_los > 180) // vza > max
				{
					m_up_los = 180;
				}
				m_min_los = m_los_list_ft(m);
				//-----------------------------
				//=========los idx===============
				uvec idx = find(m_los_list_ft == m_min_los);
				if (idx.n_elem == 0)
				{
					std::cout << "[Error] cannot find the min_los: " << m_min_los << endl;
					return 1;
				}
				los_idx = idx(0);

				//-----------------------------
				idx_tile = find(sza_mat_v <= m_up_sza && sza_mat_v >= m_dw_sza && vza_mat_v <= m_up_vza
					&& vza_mat_v >= m_dw_vza && los_mat_v <= m_up_los && los_mat_v >= m_dw_los && flag_mat_v == 1);
				if (idx_tile.n_elem < 5) continue;
				//-----------------------------

				//LOS_indx
				st = los_idx * idx_filter_los;
				ed = st + idx_filter_los - 1;
				//(1)dw_SZA_dw_VZA_LOS_indx
				fmat lut_ds_dv_l = lut_ds_dv.rows(span(st, ed));
				//(2)dw_SZA_up_VZA_LOS_indx
				fmat lut_ds_uv_l = lut_ds_uv.rows(span(st, ed));
				//(3)up_SZA_dw_VZA_LOS_indx
				fmat lut_us_dv_l = lut_us_dv.rows(span(st, ed));
				//(4)up_SZA_up_VZA_LOS_indx
				fmat lut_us_uv_l = lut_us_uv.rows(span(st, ed));

				//==================================================
				for (uword n = 0; n < 5; n++)
				{
					m_dw_dem = 1 * n;
					m_up_dem = 1 * (n + 1);
					if (n == 4) // dem > max
					{
						m_up_dem = 5.9;
					}
					//-------------------------
					//==========find up_dem and dw_dem idx===============
					idx = find(m_dem_list == m_up_dem);
					if (idx.n_elem == 0)
					{
						std::cout << "[Error] cannot find the up_dem: " << m_up_dem << endl;
						return 1;
					}
					up_dem_idx = idx(0);

					idx = find(m_dem_list == m_dw_dem);
					if (idx.n_elem == 0)
					{
						std::cout << "[Error] cannot find the dw_dem: " << m_dw_dem << endl;
						return 1;
					}
					dw_dem_idx = idx(0);

					//-----------------------------
					//�ж����ڼ���Ŀ�
					idx_tile = find(sza_mat_v >= m_dw_sza && sza_mat_v <= m_up_sza && vza_mat_v >= m_dw_vza 
						&& vza_mat_v <= m_up_vza && los_mat_v >= m_dw_los && los_mat_v <= m_up_los
						&& dem_mat_v <= m_up_dem && dem_mat_v >= m_dw_dem && flag_mat_v == 1 && toa_rad_mat_b6_v > 0 && toa_rad_mat_b7_v > 0);
					//if (idx_tile.n_elem == 0) continue;
					if (idx_tile.n_elem < 5) continue; //�ĳ�С��5��������2������

					//-----------------------------
					//
					st = dw_dem_idx * idx_filter_dem;
					ed = st + idx_filter_dem * (up_dem_idx - dw_dem_idx + 1) - 1;

					//(1)dw_SZA_dw_VZA_LOS_dem_indx
					fmat lut_ds_dv_ld = lut_ds_dv_l.rows(span(st, ed));
					//(2)dw_SZA_up_VZA_LOS_dem_indx
					fmat lut_ds_uv_ld = lut_ds_uv_l.rows(span(st, ed));
					//(3)up_SZA_dw_VZA_LOS_dem_indx
					fmat lut_us_dv_ld = lut_us_dv_l.rows(span(st, ed));
					//(4)up_SZA_up_VZA_LOS_dem_indx
					fmat lut_us_uv_ld = lut_us_uv_l.rows(span(st, ed));

					//�ϲ���
					fmat lut = lut_ds_dv_ld;
					lut.insert_rows(lut.n_rows, lut_ds_uv_ld);
					lut.insert_rows(lut.n_rows, lut_us_dv_ld);
					lut.insert_rows(lut.n_rows, lut_us_uv_ld);

					////�Բ��ұ����зֿ�
					//=============��һ�ηֿ�==================================================================
					//��Ӱ����зֿ�
					fvec dem_sub_v = dem_mat_v(idx_tile);
					fvec sza_sub_v = sza_mat_v(idx_tile);
					fvec vza_sub_v = vza_mat_v(idx_tile);

					fvec toa_rad_b1_sub_v = toa_rad_mat_b1_v(idx_tile);
					fvec toa_rad_b2_sub_v = toa_rad_mat_b2_v(idx_tile);
				    fvec toa_rad_b3_sub_v = toa_rad_mat_b3_v(idx_tile);
					fvec toa_rad_b4_sub_v = toa_rad_mat_b4_v(idx_tile);
					fvec toa_rad_b5_sub_v = toa_rad_mat_b5_v(idx_tile);
					fvec toa_rad_b6_sub_v = toa_rad_mat_b6_v(idx_tile);
					fvec toa_rad_b7_sub_v = toa_rad_mat_b7_v(idx_tile);

					fvec toa_rad_ndsi_sub_v = (toa_rad_b4_sub_v / 593.84 - toa_rad_b6_sub_v / 76.53) / (toa_rad_b4_sub_v / 593.84 + toa_rad_b6_sub_v / 76.53);

					fvec band1_ref_sub_v = band1_ref_mat_v(idx_tile);
					fvec band2_ref_sub_v = band2_ref_mat_v(idx_tile);
					fvec band3_ref_sub_v = band3_ref_mat_v(idx_tile);
					fvec band4_ref_sub_v = band4_ref_mat_v(idx_tile);
					fvec band5_ref_sub_v = band5_ref_mat_v(idx_tile);
					fvec band6_ref_sub_v = band6_ref_mat_v(idx_tile);
					fvec band7_ref_sub_v = band7_ref_mat_v(idx_tile);
					fvec sw_alb_sub_v = sw_alb_mat_v(idx_tile);
					fvec vis_alb_sub_v = vis_alb_mat_v(idx_tile);

					uword nelem_tiles = toa_rad_b3_sub_v.n_elem;

					fvec swdr_sub_v = zeros<fvec>(nelem_tiles) - 1.0;
					fvec sw_dir_sub_v = zeros<fvec>(nelem_tiles) - 1.0;
					fvec par_sub_v = zeros<fvec>(nelem_tiles) - 1.0;
					fvec pardir_sub_v = zeros<fvec>(nelem_tiles) - 1.0;
					fvec uva_sub_v = zeros<fvec>(nelem_tiles) - 1.0;
					fvec uvb_sub_v = zeros<fvec>(nelem_tiles) - 1.0;
					fvec toa_up_flux_sub_v = zeros<fvec>(nelem_tiles) - 1.0;
					fvec rho_sub_v = zeros<fvec>(nelem_tiles) - 1.0;

					////================================================================================================================
					
					//���ݻ�ѩָ���ڶ��ηֿ�
					float lut_diff_max = 1;
					float lut_diff_min = -1;
					//��һ����ѩ
					uvec idx_nosnow_1 = find(((toa_rad_b1_sub_v / 511.72 <= 0.1) || (toa_rad_b2_sub_v / 315.69 <= 0.1) || (toa_rad_b4_sub_v / 593.84 <= 0.11))
						|| (band3_ref_sub_v <= 0.3));
					if (idx_nosnow_1.n_elem != 0)
					{
						nelem_tiles = idx_nosnow_1.n_elem;

						fvec swdr_sub_v1 = zeros<fvec>(nelem_tiles) - 1.0;
						fvec sw_dir_sub_v1 = zeros<fvec>(nelem_tiles) - 1.0;
						fvec par_sub_v1 = zeros<fvec>(nelem_tiles) - 1.0;
						fvec pardir_sub_v1 = zeros<fvec>(nelem_tiles) - 1.0;
						fvec uva_sub_v1 = zeros<fvec>(nelem_tiles) - 1.0;
						fvec uvb_sub_v1 = zeros<fvec>(nelem_tiles) - 1.0;
						fvec toa_up_flux_sub_v1 = zeros<fvec>(nelem_tiles) - 1.0;
						fvec rho_sub_v1 = zeros<fvec>(nelem_tiles) - 1.0;
						//===================================================
						//��Ӱ����зֿ�
						fvec dem_sub_v1 = dem_sub_v(idx_nosnow_1);
						fvec sza_sub_v1 = sza_sub_v(idx_nosnow_1);
						fvec vza_sub_v1 = vza_sub_v(idx_nosnow_1);

						fvec toa_rad_b1_sub_v1 = toa_rad_b1_sub_v(idx_nosnow_1);
						fvec toa_rad_b3_sub_v1 = toa_rad_b3_sub_v(idx_nosnow_1);
						fvec toa_rad_b4_sub_v1 = toa_rad_b4_sub_v(idx_nosnow_1);
						fvec toa_rad_b6_sub_v1 = toa_rad_b6_sub_v(idx_nosnow_1);
						fvec toa_rad_b7_sub_v1 = toa_rad_b7_sub_v(idx_nosnow_1);

						fvec band1_ref_sub_v1 = band1_ref_sub_v(idx_nosnow_1);
						fvec band3_ref_sub_v1 = band3_ref_sub_v(idx_nosnow_1);
						fvec band4_ref_sub_v1 = band4_ref_sub_v(idx_nosnow_1);
						fvec band6_ref_sub_v1 = band6_ref_sub_v(idx_nosnow_1);
						fvec band7_ref_sub_v1 = band7_ref_sub_v(idx_nosnow_1);
						fvec sw_alb_sub_v1 = sw_alb_sub_v(idx_nosnow_1);
						fvec vis_alb_sub_v1 = vis_alb_sub_v(idx_nosnow_1);

						////��index�ӱ�����
						ok = get_SWDR(lut, sza_sub_v1, vza_sub_v1, dem_sub_v1, toa_rad_b1_sub_v1, toa_rad_b3_sub_v1,
							toa_rad_b4_sub_v1, toa_rad_b6_sub_v1, toa_rad_b7_sub_v1,
							band1_ref_sub_v1, band3_ref_sub_v1, band4_ref_sub_v1,
							band6_ref_sub_v1, band7_ref_sub_v1, sw_alb_sub_v1, vis_alb_sub_v1,
							lut_diff_max, lut_diff_min,
							swdr_sub_v1, sw_dir_sub_v1,
							par_sub_v1, pardir_sub_v1, uva_sub_v1, uvb_sub_v1, toa_up_flux_sub_v1, rho_sub_v1);
						if (ok != 0) continue; // invalid

						//�ѵõ��Ľ��д��ȥ
						swdr_sub_v(idx_nosnow_1) = swdr_sub_v1;
						sw_dir_sub_v(idx_nosnow_1) = sw_dir_sub_v1;
						par_sub_v(idx_nosnow_1) = par_sub_v1;
						pardir_sub_v(idx_nosnow_1) = pardir_sub_v1;
						uva_sub_v(idx_nosnow_1) = uva_sub_v1;
						uvb_sub_v(idx_nosnow_1) = uvb_sub_v1;
						toa_up_flux_sub_v(idx_nosnow_1) = toa_up_flux_sub_v1;
						rho_sub_v(idx_nosnow_1) = rho_sub_v1;

						//return 0;
					}

					//�ڶ�����ѩ
					uvec idx_nosnow_2 = find((toa_rad_b1_sub_v / 511.72 > 0.1) && (toa_rad_b2_sub_v / 315.69 > 0.1) && (toa_rad_b4_sub_v / 593.84 > 0.11)
						&& (band3_ref_sub_v > 0.3) && (toa_rad_ndsi_sub_v < 0.1));

					if (idx_nosnow_2.n_elem != 0)
					{
						lut_diff_max = 0.1;

						nelem_tiles = idx_nosnow_2.n_elem;

						fvec swdr_sub_v2 = zeros<fvec>(nelem_tiles) - 1.0;
						fvec sw_dir_sub_v2 = zeros<fvec>(nelem_tiles) - 1.0;
						fvec par_sub_v2 = zeros<fvec>(nelem_tiles) - 1.0;
						fvec pardir_sub_v2 = zeros<fvec>(nelem_tiles) - 1.0;
						fvec uva_sub_v2 = zeros<fvec>(nelem_tiles) - 1.0;
						fvec uvb_sub_v2 = zeros<fvec>(nelem_tiles) - 1.0;
						fvec toa_up_flux_sub_v2 = zeros<fvec>(nelem_tiles) - 1.0;
						fvec rho_sub_v2 = zeros<fvec>(nelem_tiles) - 1.0;
						//===================================================
						//��Ӱ����зֿ�
						fvec dem_sub_v2 = dem_sub_v(idx_nosnow_2);
						fvec sza_sub_v2 = sza_sub_v(idx_nosnow_2);
						fvec vza_sub_v2 = vza_sub_v(idx_nosnow_2);

						fvec toa_rad_b1_sub_v2 = toa_rad_b1_sub_v(idx_nosnow_2);
						fvec toa_rad_b3_sub_v2 = toa_rad_b3_sub_v(idx_nosnow_2);
						fvec toa_rad_b4_sub_v2 = toa_rad_b4_sub_v(idx_nosnow_2);
						fvec toa_rad_b6_sub_v2 = toa_rad_b6_sub_v(idx_nosnow_2);
						fvec toa_rad_b7_sub_v2 = toa_rad_b7_sub_v(idx_nosnow_2);

						fvec band1_ref_sub_v2 = band1_ref_sub_v(idx_nosnow_2);
						fvec band3_ref_sub_v2 = band3_ref_sub_v(idx_nosnow_2);
						fvec band4_ref_sub_v2 = band4_ref_sub_v(idx_nosnow_2);
						fvec band6_ref_sub_v2 = band6_ref_sub_v(idx_nosnow_2);
						fvec band7_ref_sub_v2 = band7_ref_sub_v(idx_nosnow_2);
						fvec sw_alb_sub_v2 = sw_alb_sub_v(idx_nosnow_2);
						fvec vis_alb_sub_v2 = vis_alb_sub_v(idx_nosnow_2);

						//���һ��LUT�ӱ�����,!!�����ٶ��һ���ܱ����ݣ����һ���Ƕ������£����ӱ��Ҳ������ñ���ʱ���������ܱ��ҿ��ñ���
						ok = get_SWDR(lut, sza_sub_v2, vza_sub_v2, dem_sub_v2, toa_rad_b1_sub_v2, toa_rad_b3_sub_v2,
							toa_rad_b4_sub_v2, toa_rad_b6_sub_v2, toa_rad_b7_sub_v2,
							band1_ref_sub_v2, band3_ref_sub_v2, band4_ref_sub_v2, 
							band6_ref_sub_v2, band7_ref_sub_v2, sw_alb_sub_v2, vis_alb_sub_v2,
							lut_diff_max, lut_diff_min,
							swdr_sub_v2, sw_dir_sub_v2,
							par_sub_v2, pardir_sub_v2, uva_sub_v2, uvb_sub_v2, toa_up_flux_sub_v2, rho_sub_v2);
						if (ok != 0) continue; // invalid

						//�ѵõ��Ľ��д��ȥ
						swdr_sub_v(idx_nosnow_2) = swdr_sub_v2;
						sw_dir_sub_v(idx_nosnow_2) = sw_dir_sub_v2;
						par_sub_v(idx_nosnow_2) = par_sub_v2;
						pardir_sub_v(idx_nosnow_2) = pardir_sub_v2;
						uva_sub_v(idx_nosnow_2) = uva_sub_v2;
						uvb_sub_v(idx_nosnow_2) = uvb_sub_v2;
						toa_up_flux_sub_v(idx_nosnow_2) = toa_up_flux_sub_v2;
						rho_sub_v(idx_nosnow_2) = rho_sub_v2;

						//return 0;
					}

					//��������ѩ (���ü���2����׼�
					uvec idx_snow_3 = find((toa_rad_b1_sub_v / 511.72 > 0.1) && (toa_rad_b2_sub_v / 315.69 > 0.1) && (toa_rad_b4_sub_v / 593.84 > 0.11)
						&& (band3_ref_sub_v > 0.3) && (toa_rad_ndsi_sub_v >= 0.1));

					if (idx_snow_3.n_elem != 0)
					{
						nelem_tiles = idx_snow_3.n_elem;

						fvec swdr_sub_v3 = zeros<fvec>(nelem_tiles) - 1.0;
						fvec sw_dir_sub_v3 = zeros<fvec>(nelem_tiles) - 1.0;
						fvec par_sub_v3 = zeros<fvec>(nelem_tiles) - 1.0;
						fvec pardir_sub_v3 = zeros<fvec>(nelem_tiles) - 1.0;
						fvec uva_sub_v3 = zeros<fvec>(nelem_tiles) - 1.0;
						fvec uvb_sub_v3 = zeros<fvec>(nelem_tiles) - 1.0;
						fvec toa_up_flux_sub_v3 = zeros<fvec>(nelem_tiles) - 1.0;
						fvec rho_sub_v3 = zeros<fvec>(nelem_tiles) - 1.0;
						//======================================================
						//��Ӱ����зֿ�
						fvec dem_sub_v3 = dem_sub_v(idx_snow_3);
						fvec sza_sub_v3 = sza_sub_v(idx_snow_3);
						fvec vza_sub_v3 = vza_sub_v(idx_snow_3);

						fvec toa_rad_b1_sub_v3 = toa_rad_b1_sub_v(idx_snow_3);
						fvec toa_rad_b3_sub_v3 = toa_rad_b3_sub_v(idx_snow_3);
						fvec toa_rad_b4_sub_v3 = toa_rad_b4_sub_v(idx_snow_3);
						fvec toa_rad_b6_sub_v3 = toa_rad_b6_sub_v(idx_snow_3);
						fvec toa_rad_b7_sub_v3 = toa_rad_b7_sub_v(idx_snow_3);

						fvec band1_ref_sub_v3 = band1_ref_sub_v(idx_snow_3);
						fvec band3_ref_sub_v3 = band3_ref_sub_v(idx_snow_3);
						fvec band4_ref_sub_v3 = band4_ref_sub_v(idx_snow_3);
						fvec band6_ref_sub_v3 = band6_ref_sub_v(idx_snow_3);
						fvec band7_ref_sub_v3 = band7_ref_sub_v(idx_snow_3);
						fvec sw_alb_sub_v3 = sw_alb_sub_v(idx_snow_3);
						fvec vis_alb_sub_v3 = vis_alb_sub_v(idx_snow_3);

						fvec toa_rad_ndsi_sub_v3 = toa_rad_ndsi_sub_v(idx_snow_3);

						//------------------------------------------------------------------
						for (int i = 0; i < 90; i++)
						{
							lut_diff_max = i * 0.01 + 0.1;
							lut_diff_min = i * 0.01 + 0.11;
							
							uvec idx_snow_3_i = find(toa_rad_ndsi_sub_v3 >= (i * 0.01 + 0.1) && toa_rad_ndsi_sub_v3 < (i * 0.01 + 0.11));
							if (idx_snow_3_i.n_elem == 0) continue;
							
							//--------------------------------------------------------------
							nelem_tiles = idx_snow_3_i.n_elem;

							fvec swdr_sub_v3_sub = zeros<fvec>(nelem_tiles) - 1.0;
							fvec sw_dir_sub_v3_sub = zeros<fvec>(nelem_tiles) - 1.0;
							fvec par_sub_v3_sub = zeros<fvec>(nelem_tiles) - 1.0;
							fvec pardir_sub_v3_sub = zeros<fvec>(nelem_tiles) - 1.0;
							fvec uva_sub_v3_sub = zeros<fvec>(nelem_tiles) - 1.0;
							fvec uvb_sub_v3_sub = zeros<fvec>(nelem_tiles) - 1.0;
							fvec toa_up_flux_sub_v3_sub = zeros<fvec>(nelem_tiles) - 1.0;
							fvec rho_sub_v3_sub = zeros<fvec>(nelem_tiles) - 1.0;
							//======================================================

							//��Ӱ����зֿ�
							fvec dem_sub_v3_sub = dem_sub_v3(idx_snow_3_i);
							fvec sza_sub_v3_sub = sza_sub_v3(idx_snow_3_i);
							fvec vza_sub_v3_sub = vza_sub_v3(idx_snow_3_i);

							fvec toa_rad_b1_sub_v3_sub = toa_rad_b1_sub_v3(idx_snow_3_i);
							fvec toa_rad_b3_sub_v3_sub = toa_rad_b3_sub_v3(idx_snow_3_i);
							fvec toa_rad_b4_sub_v3_sub = toa_rad_b4_sub_v3(idx_snow_3_i);
							fvec toa_rad_b6_sub_v3_sub = toa_rad_b6_sub_v3(idx_snow_3_i);
							fvec toa_rad_b7_sub_v3_sub = toa_rad_b7_sub_v3(idx_snow_3_i);

							fvec band1_ref_sub_v3_sub = band1_ref_sub_v3(idx_snow_3_i);
							fvec band3_ref_sub_v3_sub = band3_ref_sub_v3(idx_snow_3_i);
							fvec band4_ref_sub_v3_sub = band4_ref_sub_v3(idx_snow_3_i);
							fvec band6_ref_sub_v3_sub = band6_ref_sub_v3(idx_snow_3_i);
							fvec band7_ref_sub_v3_sub = band7_ref_sub_v3(idx_snow_3_i);
							fvec sw_alb_sub_v3_sub = sw_alb_sub_v3(idx_snow_3_i);
							fvec vis_alb_sub_v3_sub = vis_alb_sub_v3(idx_snow_3_i);

							//��index�ӱ�����
							ok = get_SWDR(lut, sza_sub_v3_sub, vza_sub_v3_sub, dem_sub_v3_sub, toa_rad_b1_sub_v3_sub, toa_rad_b3_sub_v3_sub,
								toa_rad_b4_sub_v3_sub, toa_rad_b6_sub_v3_sub, toa_rad_b7_sub_v3_sub,
								band1_ref_sub_v3_sub, band3_ref_sub_v3_sub, band4_ref_sub_v3_sub,
								band6_ref_sub_v3_sub, band7_ref_sub_v3_sub, sw_alb_sub_v3_sub, vis_alb_sub_v3_sub,
								lut_diff_max, lut_diff_min,
								swdr_sub_v3_sub, sw_dir_sub_v3_sub,
								par_sub_v3_sub, pardir_sub_v3_sub, uva_sub_v3_sub, uvb_sub_v3_sub, toa_up_flux_sub_v3_sub, rho_sub_v3_sub);
							if (ok != 0) continue; // invalid

							//�ѵõ��Ľ��д��ȥ
							swdr_sub_v3(idx_snow_3_i) = swdr_sub_v3_sub;
							sw_dir_sub_v3(idx_snow_3_i) = sw_dir_sub_v3_sub;
							par_sub_v3(idx_snow_3_i) = par_sub_v3_sub;
							pardir_sub_v3(idx_snow_3_i) = pardir_sub_v3_sub;
							uva_sub_v3(idx_snow_3_i) = uva_sub_v3_sub;
							uvb_sub_v3(idx_snow_3_i) = uvb_sub_v3_sub;
							toa_up_flux_sub_v3(idx_snow_3_i) = toa_up_flux_sub_v3_sub;
							rho_sub_v3(idx_snow_3_i) = rho_sub_v3_sub;

							//return 0;
						}

						//�ѵõ��Ľ��д��ȥ
						swdr_sub_v(idx_snow_3) = swdr_sub_v3;
						sw_dir_sub_v(idx_snow_3) = sw_dir_sub_v3;
						par_sub_v(idx_snow_3) = par_sub_v3;
						pardir_sub_v(idx_snow_3) = pardir_sub_v3;
						uva_sub_v(idx_snow_3) = uva_sub_v3;
						uvb_sub_v(idx_snow_3) = uvb_sub_v3;
						toa_up_flux_sub_v(idx_snow_3) = toa_up_flux_sub_v3;
						rho_sub_v(idx_snow_3) = rho_sub_v3;
					}

					//==================================================================================
					//���Ƕȷֿ���д��ȥ
					swdr_v(idx_tile) = swdr_sub_v;
					sw_dir_v(idx_tile) = sw_dir_sub_v;
					par_v(idx_tile) = par_sub_v;
					pardir_v(idx_tile) = pardir_sub_v;
					uva_v(idx_tile) = uva_sub_v;
					uvb_v(idx_tile) = uvb_sub_v;
					toa_up_flux_v(idx_tile) = toa_up_flux_sub_v;
					rho_v(idx_tile) = rho_sub_v;

				} // end m,LOS

			} // end m,LOS
			
		} // end j,VZA

	} // end i,SZA

	//һά���ά
	swdr_mat = reshape(swdr_v, nrows, ncols);
	sw_dir_mat = reshape(sw_dir_v, nrows, ncols);
	par_mat = reshape(par_v, nrows, ncols);
	pardir_mat = reshape(pardir_v, nrows, ncols);

	uva_mat = reshape(uva_v, nrows, ncols);
	uvb_mat = reshape(uvb_v, nrows, ncols);
	toa_up_flux_mat = reshape(toa_up_flux_v, nrows, ncols);
	rho_mat = reshape(rho_v, nrows, ncols);

	cout << "-> " << " image time: " << timer.toc() << " seconds." << endl;

	cout << "To estimate SWDR has been finished.\n";

	//***********************************************************

	Cube<short> fluxes = zeros<Cube<short>>(nrows, ncols, 10);
	fluxes.slice(0) = conv_to<Mat<short>>::from(swdr_mat * 10);
	fluxes.slice(1) = conv_to<Mat<short>>::from(sw_dir_mat * 10);
	fluxes.slice(2) = conv_to<Mat<short>>::from(par_mat * 10);
	fluxes.slice(3) = conv_to<Mat<short>>::from(pardir_mat * 10);
	fluxes.slice(4) = conv_to<Mat<short>>::from(uva_mat * 200);        //scale_factor = 0.005
	fluxes.slice(5) = conv_to<Mat<short>>::from(uvb_mat * 1000);       //scale_factor = 0.001
	fluxes.slice(6) = conv_to<Mat<short>>::from(toa_up_flux_mat * 10); //toa���з���
	fluxes.slice(7) = conv_to<Mat<short>>::from(rho_mat * 10000);      //�������淴���ʣ������ٽ�ЧӦ����
	fluxes.slice(8) = conv_to<Mat<short>>::from(sw_alb_mat * 10000);  //�ر������η����ʣ������ٽ�ЧӦ����
	fluxes.slice(9) = conv_to<Mat<short>>::from(sza_mat * 100);       //scale_factor = 0.01

	imageGeoInfo geoinfo{ input_file };
	ok = write_3d_geotif(fluxes, geoinfo, out_file);
	if (ok != 0) return 1;

	fs::path mypath{ out_file };
	if (!fs::exists(out_file))
	{
		cerr << "cannot find output file: " << out_file << endl;
		return 1;
	}

	cout << "Save SWDR MODIS to file: " << out_file << endl;
	return 0;
}

int ahi_swdr::classify_atmos(arma::fmat& toa_rad_band1_lut, arma::fmat& toa_rad_band3_lut, arma::fmat& toa_rad_band6_lut, arma::fmat& toa_rad_band7_lut,
	arma::fmat& toa_rad_band1_lut_clear, arma::fmat& toa_rad_band3_lut_clear, arma::fmat& toa_rad_band6_lut_clear, arma::fmat& toa_rad_band7_lut_clear,
	arma::fmat& toa_rad_band1_lut_cloudy, arma::fmat& toa_rad_band3_lut_cloudy, arma::fmat& toa_rad_band7_lut_cloudy)
{
	using namespace std;
	using namespace arma;

	//��պͶ���ֱ�Ӹ���ָ����index����find���죬�����VIS=20��COD=0; ������VIS=20��COD=60
	//======================================================================================
	uword idx_ds_dv_dd = 0;
	uword idx_ds_dv_ud = idx_ds_dv_dd + idx_filter_dem;
	uword idx_ds_uv_dd = idx_ds_dv_ud + idx_filter_dem;
	uword idx_ds_uv_ud = idx_ds_uv_dd + idx_filter_dem;
	uword idx_us_dv_dd = idx_ds_uv_ud + idx_filter_dem;
	uword idx_us_dv_ud = idx_us_dv_dd + idx_filter_dem;
	uword idx_us_uv_dd = idx_us_dv_ud + idx_filter_dem;
	uword idx_us_uv_ud = idx_us_uv_dd + idx_filter_dem;

	//---------�ж���գ����Ӿ�20kmΪ��׼-------------------
	//��պͶ���ֱ�Ӹ���ָ����index����find���죬�����VIS=20��COD=0; ������VIS=20��COD=60
	//--MODIS--//
	//uword idx_clear_st = 36; 
	//uword idx_clear_ed = 41;
	//--FY-3D--//
	uword idx_clear_st = 72;
	uword idx_clear_ed = 77;

	//band1
	fmat toa_rad_b1_clear = mean(toa_rad_band1_lut.rows(span(idx_us_uv_ud + idx_clear_st, idx_us_uv_ud + idx_clear_ed)));
	fmat band1_clear_sub = toa_rad_band1_lut_clear.rows(span(idx_us_uv_ud, idx_us_uv_ud + idx_filter_dem - 1));
	band1_clear_sub.each_row() = toa_rad_b1_clear;
	toa_rad_band1_lut_clear.rows(span(idx_us_uv_ud, idx_us_uv_ud + idx_filter_dem - 1)) = band1_clear_sub;

	toa_rad_b1_clear = mean(toa_rad_band1_lut.rows(span(idx_us_uv_dd + idx_clear_st, idx_us_uv_dd + idx_clear_ed)));
	band1_clear_sub = toa_rad_band1_lut_clear.rows(span(idx_us_uv_dd, idx_us_uv_dd + idx_filter_dem - 1));
	band1_clear_sub.each_row() = toa_rad_b1_clear;
	toa_rad_band1_lut_clear.rows(span(idx_us_uv_dd, idx_us_uv_dd + idx_filter_dem - 1)) = band1_clear_sub;

	toa_rad_b1_clear = mean(toa_rad_band1_lut.rows(span(idx_us_dv_ud + idx_clear_st, idx_us_dv_ud + idx_clear_ed)));
	band1_clear_sub = toa_rad_band1_lut_clear.rows(span(idx_us_dv_ud, idx_us_dv_ud + idx_filter_dem - 1));
	band1_clear_sub.each_row() = toa_rad_b1_clear;
	toa_rad_band1_lut_clear.rows(span(idx_us_dv_ud, idx_us_dv_ud + idx_filter_dem - 1)) = band1_clear_sub;

	toa_rad_b1_clear = mean(toa_rad_band1_lut.rows(span(idx_us_dv_dd + idx_clear_st, idx_us_dv_dd + idx_clear_ed)));
	band1_clear_sub = toa_rad_band1_lut_clear.rows(span(idx_us_dv_dd, idx_us_dv_dd + idx_filter_dem - 1));
	band1_clear_sub.each_row() = toa_rad_b1_clear;
	toa_rad_band1_lut_clear.rows(span(idx_us_dv_dd, idx_us_dv_dd + idx_filter_dem - 1)) = band1_clear_sub;

	toa_rad_b1_clear = mean(toa_rad_band1_lut.rows(span(idx_ds_uv_ud + idx_clear_st, idx_ds_uv_ud + idx_clear_ed)));
	band1_clear_sub = toa_rad_band1_lut_clear.rows(span(idx_ds_uv_ud, idx_ds_uv_ud + idx_filter_dem - 1));
	band1_clear_sub.each_row() = toa_rad_b1_clear;
	toa_rad_band1_lut_clear.rows(span(idx_ds_uv_ud, idx_ds_uv_ud + idx_filter_dem - 1)) = band1_clear_sub;

	toa_rad_b1_clear = mean(toa_rad_band1_lut.rows(span(idx_ds_uv_dd + idx_clear_st, idx_ds_uv_dd + idx_clear_ed)));
	band1_clear_sub = toa_rad_band1_lut_clear.rows(span(idx_ds_uv_dd, idx_ds_uv_dd + idx_filter_dem - 1));
	band1_clear_sub.each_row() = toa_rad_b1_clear;
	toa_rad_band1_lut_clear.rows(span(idx_ds_uv_dd, idx_ds_uv_dd + idx_filter_dem - 1)) = band1_clear_sub;

	toa_rad_b1_clear = mean(toa_rad_band1_lut.rows(span(idx_ds_dv_ud + idx_clear_st, idx_ds_dv_ud + idx_clear_ed)));
	band1_clear_sub = toa_rad_band1_lut_clear.rows(span(idx_ds_dv_ud, idx_ds_dv_ud + idx_filter_dem - 1));
	band1_clear_sub.each_row() = toa_rad_b1_clear;
	toa_rad_band1_lut_clear.rows(span(idx_ds_dv_ud, idx_ds_dv_ud + idx_filter_dem - 1)) = band1_clear_sub;

	toa_rad_b1_clear = mean(toa_rad_band1_lut.rows(span(idx_ds_dv_dd + idx_clear_st, idx_ds_dv_dd + idx_clear_ed)));
	band1_clear_sub = toa_rad_band1_lut_clear.rows(span(idx_ds_dv_dd, idx_ds_dv_dd + idx_filter_dem - 1));
	band1_clear_sub.each_row() = toa_rad_b1_clear;
	toa_rad_band1_lut_clear.rows(span(idx_ds_dv_dd, idx_ds_dv_dd + idx_filter_dem - 1)) = band1_clear_sub;

	//band3
	fmat toa_rad_b3_clear = mean(toa_rad_band3_lut.rows(span(idx_us_uv_ud + idx_clear_st, idx_us_uv_ud + idx_clear_ed)));
	fmat band3_clear_sub = toa_rad_band3_lut_clear.rows(span(idx_us_uv_ud, idx_us_uv_ud + idx_filter_dem - 1));
	band3_clear_sub.each_row() = toa_rad_b3_clear;
	toa_rad_band3_lut_clear.rows(span(idx_us_uv_ud, idx_us_uv_ud + idx_filter_dem - 1)) = band3_clear_sub;

	toa_rad_b3_clear = mean(toa_rad_band3_lut.rows(span(idx_us_uv_dd + idx_clear_st, idx_us_uv_dd + idx_clear_ed)));
	band3_clear_sub = toa_rad_band3_lut_clear.rows(span(idx_us_uv_dd, idx_us_uv_dd + idx_filter_dem - 1));
	band3_clear_sub.each_row() = toa_rad_b3_clear;
	toa_rad_band3_lut_clear.rows(span(idx_us_uv_dd, idx_us_uv_dd + idx_filter_dem - 1)) = band3_clear_sub;

	toa_rad_b3_clear = mean(toa_rad_band3_lut.rows(span(idx_us_dv_ud + idx_clear_st, idx_us_dv_ud + idx_clear_ed)));
	band3_clear_sub = toa_rad_band3_lut_clear.rows(span(idx_us_dv_ud, idx_us_dv_ud + idx_filter_dem - 1));
	band3_clear_sub.each_row() = toa_rad_b3_clear;
	toa_rad_band3_lut_clear.rows(span(idx_us_dv_ud, idx_us_dv_ud + idx_filter_dem - 1)) = band3_clear_sub;

	toa_rad_b3_clear = mean(toa_rad_band3_lut.rows(span(idx_us_dv_dd + idx_clear_st, idx_us_dv_dd + idx_clear_ed)));
	band3_clear_sub = toa_rad_band3_lut_clear.rows(span(idx_us_dv_dd, idx_us_dv_dd + idx_filter_dem - 1));
	band3_clear_sub.each_row() = toa_rad_b3_clear;
	toa_rad_band3_lut_clear.rows(span(idx_us_dv_dd, idx_us_dv_dd + idx_filter_dem - 1)) = band3_clear_sub;

	toa_rad_b3_clear = mean(toa_rad_band3_lut.rows(span(idx_ds_uv_ud + idx_clear_st, idx_ds_uv_ud + idx_clear_ed)));
	band3_clear_sub = toa_rad_band3_lut_clear.rows(span(idx_ds_uv_ud, idx_ds_uv_ud + idx_filter_dem - 1));
	band3_clear_sub.each_row() = toa_rad_b3_clear;
	toa_rad_band3_lut_clear.rows(span(idx_ds_uv_ud, idx_ds_uv_ud + idx_filter_dem - 1)) = band3_clear_sub;

	toa_rad_b3_clear = mean(toa_rad_band3_lut.rows(span(idx_ds_uv_dd + idx_clear_st, idx_ds_uv_dd + idx_clear_ed)));
	band3_clear_sub = toa_rad_band3_lut_clear.rows(span(idx_ds_uv_dd, idx_ds_uv_dd + idx_filter_dem - 1));
	band3_clear_sub.each_row() = toa_rad_b3_clear;
	toa_rad_band3_lut_clear.rows(span(idx_ds_uv_dd, idx_ds_uv_dd + idx_filter_dem - 1)) = band3_clear_sub;

	toa_rad_b3_clear = mean(toa_rad_band3_lut.rows(span(idx_ds_dv_ud + idx_clear_st, idx_ds_dv_ud + idx_clear_ed)));
	band3_clear_sub = toa_rad_band3_lut_clear.rows(span(idx_ds_dv_ud, idx_ds_dv_ud + idx_filter_dem - 1));
	band3_clear_sub.each_row() = toa_rad_b3_clear;
	toa_rad_band3_lut_clear.rows(span(idx_ds_dv_ud, idx_ds_dv_ud + idx_filter_dem - 1)) = band3_clear_sub;

	toa_rad_b3_clear = mean(toa_rad_band3_lut.rows(span(idx_ds_dv_dd + idx_clear_st, idx_ds_dv_dd + idx_clear_ed)));
	band3_clear_sub = toa_rad_band3_lut_clear.rows(span(idx_ds_dv_dd, idx_ds_dv_dd + idx_filter_dem - 1));
	band3_clear_sub.each_row() = toa_rad_b3_clear;
	toa_rad_band3_lut_clear.rows(span(idx_ds_dv_dd, idx_ds_dv_dd + idx_filter_dem - 1)) = band3_clear_sub;

	//band6
	fmat toa_rad_b6_clear = mean(toa_rad_band6_lut.rows(span(idx_us_uv_ud + idx_clear_st, idx_us_uv_ud + idx_clear_ed)));
	fmat band6_clear_sub = toa_rad_band6_lut_clear.rows(span(idx_us_uv_ud, idx_us_uv_ud + idx_filter_dem - 1));
	band6_clear_sub.each_row() = toa_rad_b6_clear;
	toa_rad_band6_lut_clear.rows(span(idx_us_uv_ud, idx_us_uv_ud + idx_filter_dem - 1)) = band6_clear_sub;

	toa_rad_b6_clear = mean(toa_rad_band6_lut.rows(span(idx_us_uv_dd + idx_clear_st, idx_us_uv_dd + idx_clear_ed)));
	band6_clear_sub = toa_rad_band6_lut_clear.rows(span(idx_us_uv_dd, idx_us_uv_dd + idx_filter_dem - 1));
	band6_clear_sub.each_row() = toa_rad_b6_clear;
	toa_rad_band6_lut_clear.rows(span(idx_us_uv_dd, idx_us_uv_dd + idx_filter_dem - 1)) = band6_clear_sub;

	toa_rad_b6_clear = mean(toa_rad_band6_lut.rows(span(idx_us_dv_ud + idx_clear_st, idx_us_dv_ud + idx_clear_ed)));
	band6_clear_sub = toa_rad_band6_lut_clear.rows(span(idx_us_dv_ud, idx_us_dv_ud + idx_filter_dem - 1));
	band6_clear_sub.each_row() = toa_rad_b6_clear;
	toa_rad_band6_lut_clear.rows(span(idx_us_dv_ud, idx_us_dv_ud + idx_filter_dem - 1)) = band6_clear_sub;

	toa_rad_b6_clear = mean(toa_rad_band6_lut.rows(span(idx_us_dv_dd + idx_clear_st, idx_us_dv_dd + idx_clear_ed)));
	band6_clear_sub = toa_rad_band6_lut_clear.rows(span(idx_us_dv_dd, idx_us_dv_dd + idx_filter_dem - 1));
	band6_clear_sub.each_row() = toa_rad_b6_clear;
	toa_rad_band6_lut_clear.rows(span(idx_us_dv_dd, idx_us_dv_dd + idx_filter_dem - 1)) = band6_clear_sub;

	toa_rad_b6_clear = mean(toa_rad_band6_lut.rows(span(idx_ds_uv_ud + idx_clear_st, idx_ds_uv_ud + idx_clear_ed)));
	band6_clear_sub = toa_rad_band6_lut_clear.rows(span(idx_ds_uv_ud, idx_ds_uv_ud + idx_filter_dem - 1));
	band6_clear_sub.each_row() = toa_rad_b6_clear;
	toa_rad_band6_lut_clear.rows(span(idx_ds_uv_ud, idx_ds_uv_ud + idx_filter_dem - 1)) = band6_clear_sub;

	toa_rad_b6_clear = mean(toa_rad_band6_lut.rows(span(idx_ds_uv_dd + idx_clear_st, idx_ds_uv_dd + idx_clear_ed)));
	band6_clear_sub = toa_rad_band6_lut_clear.rows(span(idx_ds_uv_dd, idx_ds_uv_dd + idx_filter_dem - 1));
	band6_clear_sub.each_row() = toa_rad_b6_clear;
	toa_rad_band6_lut_clear.rows(span(idx_ds_uv_dd, idx_ds_uv_dd + idx_filter_dem - 1)) = band6_clear_sub;

	toa_rad_b6_clear = mean(toa_rad_band6_lut.rows(span(idx_ds_dv_ud + idx_clear_st, idx_ds_dv_ud + idx_clear_ed)));
	band6_clear_sub = toa_rad_band6_lut_clear.rows(span(idx_ds_dv_ud, idx_ds_dv_ud + idx_filter_dem - 1));
	band6_clear_sub.each_row() = toa_rad_b6_clear;
	toa_rad_band6_lut_clear.rows(span(idx_ds_dv_ud, idx_ds_dv_ud + idx_filter_dem - 1)) = band6_clear_sub;

	toa_rad_b6_clear = mean(toa_rad_band6_lut.rows(span(idx_ds_dv_dd + idx_clear_st, idx_ds_dv_dd + idx_clear_ed)));
	band6_clear_sub = toa_rad_band6_lut_clear.rows(span(idx_ds_dv_dd, idx_ds_dv_dd + idx_filter_dem - 1));
	band6_clear_sub.each_row() = toa_rad_b6_clear;
	toa_rad_band6_lut_clear.rows(span(idx_ds_dv_dd, idx_ds_dv_dd + idx_filter_dem - 1)) = band6_clear_sub;

	//band7
	fmat toa_rad_b7_clear = mean(toa_rad_band7_lut.rows(span(idx_us_uv_ud + idx_clear_st, idx_us_uv_ud + idx_clear_ed)));
	fmat band7_clear_sub = toa_rad_band7_lut_clear.rows(span(idx_us_uv_ud, idx_us_uv_ud + idx_filter_dem - 1));
	band7_clear_sub.each_row() = toa_rad_b7_clear;
	toa_rad_band7_lut_clear.rows(span(idx_us_uv_ud, idx_us_uv_ud + idx_filter_dem - 1)) = band7_clear_sub;

	toa_rad_b7_clear = mean(toa_rad_band7_lut.rows(span(idx_us_uv_dd + idx_clear_st, idx_us_uv_dd + idx_clear_ed)));
	band7_clear_sub = toa_rad_band7_lut_clear.rows(span(idx_us_uv_dd, idx_us_uv_dd + idx_filter_dem - 1));
	band7_clear_sub.each_row() = toa_rad_b7_clear;
	toa_rad_band7_lut_clear.rows(span(idx_us_uv_dd, idx_us_uv_dd + idx_filter_dem - 1)) = band7_clear_sub;

	toa_rad_b7_clear = mean(toa_rad_band7_lut.rows(span(idx_us_dv_ud + idx_clear_st, idx_us_dv_ud + idx_clear_ed)));
	band7_clear_sub = toa_rad_band7_lut_clear.rows(span(idx_us_dv_ud, idx_us_dv_ud + idx_filter_dem - 1));
	band7_clear_sub.each_row() = toa_rad_b7_clear;
	toa_rad_band7_lut_clear.rows(span(idx_us_dv_ud, idx_us_dv_ud + idx_filter_dem - 1)) = band7_clear_sub;

	toa_rad_b7_clear = mean(toa_rad_band7_lut.rows(span(idx_us_dv_dd + idx_clear_st, idx_us_dv_dd + idx_clear_ed)));
	band7_clear_sub = toa_rad_band7_lut_clear.rows(span(idx_us_dv_dd, idx_us_dv_dd + idx_filter_dem - 1));
	band7_clear_sub.each_row() = toa_rad_b7_clear;
	toa_rad_band7_lut_clear.rows(span(idx_us_dv_dd, idx_us_dv_dd + idx_filter_dem - 1)) = band7_clear_sub;

	toa_rad_b7_clear = mean(toa_rad_band7_lut.rows(span(idx_ds_uv_ud + idx_clear_st, idx_ds_uv_ud + idx_clear_ed)));
	band7_clear_sub = toa_rad_band7_lut_clear.rows(span(idx_ds_uv_ud, idx_ds_uv_ud + idx_filter_dem - 1));
	band7_clear_sub.each_row() = toa_rad_b7_clear;
	toa_rad_band7_lut_clear.rows(span(idx_ds_uv_ud, idx_ds_uv_ud + idx_filter_dem - 1)) = band7_clear_sub;

	toa_rad_b7_clear = mean(toa_rad_band7_lut.rows(span(idx_ds_uv_dd + idx_clear_st, idx_ds_uv_dd + idx_clear_ed)));
	band7_clear_sub = toa_rad_band7_lut_clear.rows(span(idx_ds_uv_dd, idx_ds_uv_dd + idx_filter_dem - 1));
	band7_clear_sub.each_row() = toa_rad_b7_clear;
	toa_rad_band7_lut_clear.rows(span(idx_ds_uv_dd, idx_ds_uv_dd + idx_filter_dem - 1)) = band7_clear_sub;

	toa_rad_b7_clear = mean(toa_rad_band7_lut.rows(span(idx_ds_dv_ud + idx_clear_st, idx_ds_dv_ud + idx_clear_ed)));
	band7_clear_sub = toa_rad_band7_lut_clear.rows(span(idx_ds_dv_ud, idx_ds_dv_ud + idx_filter_dem - 1));
	band7_clear_sub.each_row() = toa_rad_b7_clear;
	toa_rad_band7_lut_clear.rows(span(idx_ds_dv_ud, idx_ds_dv_ud + idx_filter_dem - 1)) = band7_clear_sub;

	toa_rad_b7_clear = mean(toa_rad_band7_lut.rows(span(idx_ds_dv_dd + idx_clear_st, idx_ds_dv_dd + idx_clear_ed)));
	band7_clear_sub = toa_rad_band7_lut_clear.rows(span(idx_ds_dv_dd, idx_ds_dv_dd + idx_filter_dem - 1));
	band7_clear_sub.each_row() = toa_rad_b7_clear;
	toa_rad_band7_lut_clear.rows(span(idx_ds_dv_dd, idx_ds_dv_dd + idx_filter_dem - 1)) = band7_clear_sub;

	//===================================================================
	//����
	//uvec idx_cloudy = { 207,221,235 };//modis
	uvec idx_cloudy = { 243,257,271 }; //FY-3D��������յ�vis
	//------------------------------------------

	//band1
	fmat toa_rad_b1_cloudy = mean(toa_rad_band1_lut.rows(idx_us_uv_ud + idx_cloudy));
	fmat band1_cloudy_sub = toa_rad_band1_lut_cloudy.rows(span(idx_us_uv_ud, idx_us_uv_ud + idx_filter_dem - 1));
	band1_cloudy_sub.each_row() = toa_rad_b1_cloudy;
	toa_rad_band1_lut_cloudy.rows(span(idx_us_uv_ud, idx_us_uv_ud + idx_filter_dem - 1)) = band1_cloudy_sub;

	toa_rad_b1_cloudy = mean(toa_rad_band1_lut.rows(idx_us_uv_dd + idx_cloudy));
	band1_cloudy_sub = toa_rad_band1_lut_cloudy.rows(span(idx_us_uv_dd, idx_us_uv_dd + idx_filter_dem - 1));
	band1_cloudy_sub.each_row() = toa_rad_b1_cloudy;
	toa_rad_band1_lut_cloudy.rows(span(idx_us_uv_dd, idx_us_uv_dd + idx_filter_dem - 1)) = band1_cloudy_sub;

	toa_rad_b1_cloudy = mean(toa_rad_band1_lut.rows(idx_us_dv_ud + idx_cloudy));
	band1_cloudy_sub = toa_rad_band1_lut_cloudy.rows(span(idx_us_dv_ud, idx_us_dv_ud + idx_filter_dem - 1));
	band1_cloudy_sub.each_row() = toa_rad_b1_cloudy;
	toa_rad_band1_lut_cloudy.rows(span(idx_us_dv_ud, idx_us_dv_ud + idx_filter_dem - 1)) = band1_cloudy_sub;

	toa_rad_b1_cloudy = mean(toa_rad_band1_lut.rows(idx_us_dv_dd + idx_cloudy));
	band1_cloudy_sub = toa_rad_band1_lut_cloudy.rows(span(idx_us_dv_dd, idx_us_dv_dd + idx_filter_dem - 1));
	band1_cloudy_sub.each_row() = toa_rad_b1_cloudy;
	toa_rad_band1_lut_cloudy.rows(span(idx_us_dv_dd, idx_us_dv_dd + idx_filter_dem - 1)) = band1_cloudy_sub;

	toa_rad_b1_cloudy = mean(toa_rad_band1_lut.rows(idx_ds_uv_ud + idx_cloudy));
	band1_cloudy_sub = toa_rad_band1_lut_cloudy.rows(span(idx_ds_uv_ud, idx_ds_uv_ud + idx_filter_dem - 1));
	band1_cloudy_sub.each_row() = toa_rad_b1_cloudy;
	toa_rad_band1_lut_cloudy.rows(span(idx_ds_uv_ud, idx_ds_uv_ud + idx_filter_dem - 1)) = band1_cloudy_sub;

	toa_rad_b1_cloudy = mean(toa_rad_band1_lut.rows(idx_ds_uv_dd + idx_cloudy));
	band1_cloudy_sub = toa_rad_band1_lut_cloudy.rows(span(idx_ds_uv_dd, idx_ds_uv_dd + idx_filter_dem - 1));
	band1_cloudy_sub.each_row() = toa_rad_b1_cloudy;
	toa_rad_band1_lut_cloudy.rows(span(idx_ds_uv_dd, idx_ds_uv_dd + idx_filter_dem - 1)) = band1_cloudy_sub;

	toa_rad_b1_cloudy = mean(toa_rad_band1_lut.rows(idx_ds_dv_ud + idx_cloudy));
	band1_cloudy_sub = toa_rad_band1_lut_cloudy.rows(span(idx_ds_dv_ud, idx_ds_dv_ud + idx_filter_dem - 1));
	band1_cloudy_sub.each_row() = toa_rad_b1_cloudy;
	toa_rad_band1_lut_cloudy.rows(span(idx_ds_dv_ud, idx_ds_dv_ud + idx_filter_dem - 1)) = band1_cloudy_sub;

	toa_rad_b1_cloudy = mean(toa_rad_band1_lut.rows(idx_ds_dv_dd + idx_cloudy));
	band1_cloudy_sub = toa_rad_band1_lut_cloudy.rows(span(idx_ds_dv_dd, idx_ds_dv_dd + idx_filter_dem - 1));
	band1_cloudy_sub.each_row() = toa_rad_b1_cloudy;
	toa_rad_band1_lut_cloudy.rows(span(idx_ds_dv_dd, idx_ds_dv_dd + idx_filter_dem - 1)) = band1_cloudy_sub;

	//band3
	fmat toa_rad_b3_cloudy = mean(toa_rad_band3_lut.rows(idx_us_uv_ud + idx_cloudy));
	fmat band3_cloudy_sub = toa_rad_band3_lut_cloudy.rows(span(idx_us_uv_ud, idx_us_uv_ud + idx_filter_dem - 1));
	band3_cloudy_sub.each_row() = toa_rad_b3_cloudy;
	toa_rad_band3_lut_cloudy.rows(span(idx_us_uv_ud, idx_us_uv_ud + idx_filter_dem - 1)) = band3_cloudy_sub;

	toa_rad_b3_cloudy = mean(toa_rad_band3_lut.rows(idx_us_uv_dd + idx_cloudy));
	band3_cloudy_sub = toa_rad_band3_lut_cloudy.rows(span(idx_us_uv_dd, idx_us_uv_dd + idx_filter_dem - 1));
	band3_cloudy_sub.each_row() = toa_rad_b3_cloudy;
	toa_rad_band3_lut_cloudy.rows(span(idx_us_uv_dd, idx_us_uv_dd + idx_filter_dem - 1)) = band3_cloudy_sub;

	toa_rad_b3_cloudy = mean(toa_rad_band3_lut.rows(idx_us_dv_ud + idx_cloudy));
	band3_cloudy_sub = toa_rad_band3_lut_cloudy.rows(span(idx_us_dv_ud, idx_us_dv_ud + idx_filter_dem - 1));
	band3_cloudy_sub.each_row() = toa_rad_b3_cloudy;
	toa_rad_band3_lut_cloudy.rows(span(idx_us_dv_ud, idx_us_dv_ud + idx_filter_dem - 1)) = band3_cloudy_sub;

	toa_rad_b3_cloudy = mean(toa_rad_band3_lut.rows(idx_us_dv_dd + idx_cloudy));
	band3_cloudy_sub = toa_rad_band3_lut_cloudy.rows(span(idx_us_dv_dd, idx_us_dv_dd + idx_filter_dem - 1));
	band3_cloudy_sub.each_row() = toa_rad_b3_cloudy;
	toa_rad_band3_lut_cloudy.rows(span(idx_us_dv_dd, idx_us_dv_dd + idx_filter_dem - 1)) = band3_cloudy_sub;

	toa_rad_b3_cloudy = mean(toa_rad_band3_lut.rows(idx_ds_uv_ud + idx_cloudy));
	band3_cloudy_sub = toa_rad_band3_lut_cloudy.rows(span(idx_ds_uv_ud, idx_ds_uv_ud + idx_filter_dem - 1));
	band3_cloudy_sub.each_row() = toa_rad_b3_cloudy;
	toa_rad_band3_lut_cloudy.rows(span(idx_ds_uv_ud, idx_ds_uv_ud + idx_filter_dem - 1)) = band3_cloudy_sub;

	toa_rad_b3_cloudy = mean(toa_rad_band3_lut.rows(idx_ds_uv_dd + idx_cloudy));
	band3_cloudy_sub = toa_rad_band3_lut_cloudy.rows(span(idx_ds_uv_dd, idx_ds_uv_dd + idx_filter_dem - 1));
	band3_cloudy_sub.each_row() = toa_rad_b3_cloudy;
	toa_rad_band3_lut_cloudy.rows(span(idx_ds_uv_dd, idx_ds_uv_dd + idx_filter_dem - 1)) = band3_cloudy_sub;

	toa_rad_b3_cloudy = mean(toa_rad_band3_lut.rows(idx_ds_dv_ud + idx_cloudy));
	band3_cloudy_sub = toa_rad_band3_lut_cloudy.rows(span(idx_ds_dv_ud, idx_ds_dv_ud + idx_filter_dem - 1));
	band3_cloudy_sub.each_row() = toa_rad_b3_cloudy;
	toa_rad_band3_lut_cloudy.rows(span(idx_ds_dv_ud, idx_ds_dv_ud + idx_filter_dem - 1)) = band3_cloudy_sub;

	toa_rad_b3_cloudy = mean(toa_rad_band3_lut.rows(idx_ds_dv_dd + idx_cloudy));
	band3_cloudy_sub = toa_rad_band3_lut_cloudy.rows(span(idx_ds_dv_dd, idx_ds_dv_dd + idx_filter_dem - 1));
	band3_cloudy_sub.each_row() = toa_rad_b3_cloudy;
	toa_rad_band3_lut_cloudy.rows(span(idx_ds_dv_dd, idx_ds_dv_dd + idx_filter_dem - 1)) = band3_cloudy_sub;

	//band7
	fmat toa_rad_b7_cloudy = mean(toa_rad_band7_lut.rows(idx_us_uv_ud + idx_cloudy));
	fmat band7_cloudy_sub = toa_rad_band7_lut_cloudy.rows(span(idx_us_uv_ud, idx_us_uv_ud + idx_filter_dem - 1));
	band7_cloudy_sub.each_row() = toa_rad_b7_cloudy;
	toa_rad_band7_lut_cloudy.rows(span(idx_us_uv_ud, idx_us_uv_ud + idx_filter_dem - 1)) = band7_cloudy_sub;

	toa_rad_b7_cloudy = mean(toa_rad_band7_lut.rows(idx_us_uv_dd + idx_cloudy));
	band7_cloudy_sub = toa_rad_band7_lut_cloudy.rows(span(idx_us_uv_dd, idx_us_uv_dd + idx_filter_dem - 1));
	band7_cloudy_sub.each_row() = toa_rad_b7_cloudy;
	toa_rad_band7_lut_cloudy.rows(span(idx_us_uv_dd, idx_us_uv_dd + idx_filter_dem - 1)) = band7_cloudy_sub;

	toa_rad_b7_cloudy = mean(toa_rad_band7_lut.rows(idx_us_dv_ud + idx_cloudy));
	band7_cloudy_sub = toa_rad_band7_lut_cloudy.rows(span(idx_us_dv_ud, idx_us_dv_ud + idx_filter_dem - 1));
	band7_cloudy_sub.each_row() = toa_rad_b7_cloudy;
	toa_rad_band7_lut_cloudy.rows(span(idx_us_dv_ud, idx_us_dv_ud + idx_filter_dem - 1)) = band7_cloudy_sub;

	toa_rad_b7_cloudy = mean(toa_rad_band7_lut.rows(idx_us_dv_dd + idx_cloudy));
	band7_cloudy_sub = toa_rad_band7_lut_cloudy.rows(span(idx_us_dv_dd, idx_us_dv_dd + idx_filter_dem - 1));
	band7_cloudy_sub.each_row() = toa_rad_b7_cloudy;
	toa_rad_band7_lut_cloudy.rows(span(idx_us_dv_dd, idx_us_dv_dd + idx_filter_dem - 1)) = band7_cloudy_sub;

	toa_rad_b7_cloudy = mean(toa_rad_band7_lut.rows(idx_ds_uv_ud + idx_cloudy));
	band7_cloudy_sub = toa_rad_band7_lut_cloudy.rows(span(idx_ds_uv_ud, idx_ds_uv_ud + idx_filter_dem - 1));
	band7_cloudy_sub.each_row() = toa_rad_b7_cloudy;
	toa_rad_band7_lut_cloudy.rows(span(idx_ds_uv_ud, idx_ds_uv_ud + idx_filter_dem - 1)) = band7_cloudy_sub;

	toa_rad_b7_cloudy = mean(toa_rad_band7_lut.rows(idx_ds_uv_dd + idx_cloudy));
	band7_cloudy_sub = toa_rad_band7_lut_cloudy.rows(span(idx_ds_uv_dd, idx_ds_uv_dd + idx_filter_dem - 1));
	band7_cloudy_sub.each_row() = toa_rad_b7_cloudy;
	toa_rad_band7_lut_cloudy.rows(span(idx_ds_uv_dd, idx_ds_uv_dd + idx_filter_dem - 1)) = band7_cloudy_sub;

	toa_rad_b7_cloudy = mean(toa_rad_band7_lut.rows(idx_ds_dv_ud + idx_cloudy));
	band7_cloudy_sub = toa_rad_band7_lut_cloudy.rows(span(idx_ds_dv_ud, idx_ds_dv_ud + idx_filter_dem - 1));
	band7_cloudy_sub.each_row() = toa_rad_b7_cloudy;
	toa_rad_band7_lut_cloudy.rows(span(idx_ds_dv_ud, idx_ds_dv_ud + idx_filter_dem - 1)) = band7_cloudy_sub;

	toa_rad_b7_cloudy = mean(toa_rad_band7_lut.rows(idx_ds_dv_dd + idx_cloudy));
	band7_cloudy_sub = toa_rad_band7_lut_cloudy.rows(span(idx_ds_dv_dd, idx_ds_dv_dd + idx_filter_dem - 1));
	band7_cloudy_sub.each_row() = toa_rad_b7_cloudy;
	toa_rad_band7_lut_cloudy.rows(span(idx_ds_dv_dd, idx_ds_dv_dd + idx_filter_dem - 1)) = band7_cloudy_sub;

	return 0;
}


int ahi_swdr::get_SWDR(arma::fmat lut, arma::fvec& sza_sub_v, arma::fvec& vza_sub_v, arma::fvec& dem_sub_v,
	arma::fvec& toa_rad_b1_sub_v, arma::fvec& toa_rad_b3_sub_v, arma::fvec& toa_rad_b4_sub_v, arma::fvec& toa_rad_b6_sub_v, arma::fvec& toa_rad_b7_sub_v,
	arma::fvec& band1_ref_sub_v, arma::fvec& band3_ref_sub_v, arma::fvec& band4_ref_sub_v, arma::fvec& band6_ref_sub_v, arma::fvec& band7_ref_sub_v, 
	arma::fvec& sw_albedo_sub_v, arma::fvec& vis_albedo_sub_v,
	float lut_diff_max, float lut_diff_min,
	arma::fvec& derived_swdr, arma::fvec& derived_dir, arma::fvec& derived_par, arma::fvec& derived_pardir,
	arma::fvec& derived_uva, arma::fvec& derived_uvb,
	arma::fvec& derived_toa_up_flux, arma::fvec& derived_rho)
{
	using namespace std;
	using namespace arma;

	//======================================================================================
	uword idx_ds_dv_dd = 0;
	uword idx_ds_dv_ud = idx_ds_dv_dd + idx_filter_dem;
	uword idx_ds_uv_dd = idx_ds_dv_ud + idx_filter_dem;
	uword idx_ds_uv_ud = idx_ds_uv_dd + idx_filter_dem;
	uword idx_us_dv_dd = idx_ds_uv_ud + idx_filter_dem;
	uword idx_us_dv_ud = idx_us_dv_dd + idx_filter_dem;
	uword idx_us_uv_dd = idx_us_dv_ud + idx_filter_dem;
	uword idx_us_uv_ud = idx_us_uv_dd + idx_filter_dem;

	//��lutΪ�Ƕȹ��˺�Ĳ��ұ�
	//=========================================================================

	fvec COD = lut.col(4);
	//===============================================					
	//���Ƕ��и����ӱ�lut���ٰ���ѩָ���и�
	//��ȡ�ನ����Ϣ
	//-----------multiband-----------------------
	//--MODIS------------------------------------
	//const fvec i0_band1 = lut.col(5); //band1
	//const fvec rho_band1 = lut.col(6);
	//const fvec complex_var_band1 = lut.col(7);

	// //band2����

	//const fvec i0_band3 = lut.col(8); // band3
	//const fvec rho_band3 = lut.col(9);
	//const fvec complex_var_band3 = lut.col(10);

	//const fvec i0_band4 = lut.col(11); //band4
	//const fvec rho_band4 = lut.col(12);
	//const fvec complex_var_band4 = lut.col(13);

	//---------------------------------------------
	//FY-3D,��1��2���κ͵�3,4����ʱ�෴��(�������벻�䣬����ұ����ɣ�
	const fvec i0_band1 = lut.col(5); //band1
	const fvec rho_band1 = lut.col(6);
	const fvec complex_var_band1 = lut.col(7);

	//band2����

	const fvec i0_band3 = lut.col(8); // band3
	const fvec rho_band3 = lut.col(9);
	const fvec complex_var_band3 = lut.col(10);

	const fvec i0_band4 = lut.col(11); //band4
	const fvec rho_band4 = lut.col(12);
	const fvec complex_var_band4 = lut.col(13);

	//---------------------------------------------
	//band5����

	const fvec i0_band6 = lut.col(14); //band6
	const fvec rho_band6 = lut.col(15);
	const fvec complex_var_band6 = lut.col(16);

	const fvec i0_band7 = lut.col(17); //band7
	const fvec rho_band7 = lut.col(18);
	const fvec complex_var_band7 = lut.col(19);

	//================================================================
	//���ұ�������ɿ�
	const uword nrows_lut_tile = i0_band1.n_rows;
	const uword ncols_lut_tile = toa_rad_b1_sub_v.n_elem;  //toa_rad_b3_sub_v.n_elem;
	//=====toa_rad(multi bands)========
	//band1
	fmat i0_band1_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	i0_band1_tile.each_col() = i0_band1;

	fmat rho_band1_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	rho_band1_tile.each_col() = rho_band1;

	fmat complex_var_band1_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	complex_var_band1_tile.each_col() = complex_var_band1;

	//band3
	fmat i0_band3_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	i0_band3_tile.each_col() = i0_band3;

	fmat rho_band3_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	rho_band3_tile.each_col() = rho_band3;

	fmat complex_var_band3_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	complex_var_band3_tile.each_col() = complex_var_band3;

	//band4
	fmat i0_band4_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	i0_band4_tile.each_col() = i0_band4;

	fmat rho_band4_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	rho_band4_tile.each_col() = rho_band4;

	fmat complex_var_band4_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	complex_var_band4_tile.each_col() = complex_var_band4;

	//band6
	fmat i0_band6_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	i0_band6_tile.each_col() = i0_band6;

	fmat rho_band6_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	rho_band6_tile.each_col() = rho_band6;

	fmat complex_var_band6_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	complex_var_band6_tile.each_col() = complex_var_band6;

	//band7
	fmat i0_band7_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	i0_band7_tile.each_col() = i0_band7;

	fmat rho_band7_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	rho_band7_tile.each_col() = rho_band7;

	fmat complex_var_band7_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	complex_var_band7_tile.each_col() = complex_var_band7;

	//-----����ÿ�����ε�toa_radiance---------
	fmat i0x_band1 = trans(1 / band1_ref_sub_v) - rho_band1_tile.each_row();
	fmat toa_rad_band1_lut_tile = i0_band1_tile + (1 / i0x_band1) % complex_var_band1_tile;

	fmat i0x_band3 = trans(1 / band3_ref_sub_v) - rho_band3_tile.each_row();
	fmat toa_rad_band3_lut_tile = i0_band3_tile + (1 / i0x_band3) % complex_var_band3_tile;  //toa_rad_lut_tile

	fmat i0x_band4 = trans(1 / band4_ref_sub_v) - rho_band4_tile.each_row();
	fmat toa_rad_band4_lut_tile = i0_band4_tile + (1 / i0x_band4) % complex_var_band4_tile;

	fmat i0x_band6 = trans(1 / band6_ref_sub_v) - rho_band6_tile.each_row();
	fmat toa_rad_band6_lut_tile = i0_band6_tile + (1 / i0x_band6) % complex_var_band6_tile;

	fmat i0x_band7 = trans(1 / band7_ref_sub_v) - rho_band7_tile.each_row();
	fmat toa_rad_band7_lut_tile = i0_band7_tile + (1 / i0x_band7) % complex_var_band7_tile;

	//======================================================

	//��ǰ�ӱ�������������up/dw SZA, up/dw VZA, up/dw DEM, ÿһ���ӿ��Ӧһ����գ�һ������ֵ
	fmat toa_rad_band1_lut_clear_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	fmat toa_rad_band3_lut_clear_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	fmat toa_rad_band6_lut_clear_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	fmat toa_rad_band7_lut_clear_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);

	fmat toa_rad_band1_lut_cloudy_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	fmat toa_rad_band3_lut_cloudy_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	fmat toa_rad_band7_lut_cloudy_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);


	//������պͺ��Ʋ���
	int ok1 = classify_atmos(toa_rad_band1_lut_tile, toa_rad_band3_lut_tile, toa_rad_band6_lut_tile, toa_rad_band7_lut_tile,
		toa_rad_band1_lut_clear_tile, toa_rad_band3_lut_clear_tile, toa_rad_band6_lut_clear_tile, toa_rad_band7_lut_clear_tile,
		toa_rad_band1_lut_cloudy_tile, toa_rad_band3_lut_cloudy_tile, toa_rad_band7_lut_cloudy_tile);

	//======================================================
	//���ұ���ѩָ��ֵ(��ѩָ����LUT�ֿ�ֻ��ѭ����ɣ�����3��flag��ÿ��flag�в�ͬ����ʽ
	fmat toa_rad_ndsi_lut_tile = (toa_rad_band4_lut_tile / 593.84 - toa_rad_band6_lut_tile / 76.53) / (toa_rad_band4_lut_tile / 593.84 + toa_rad_band6_lut_tile / 76.53);//��֤��

	//------------������LUT��SWDR & PAR UVA UVB TOA_albedo-------------------------
	//=======swdr=======
	const fvec f0 = lut.col(20) + lut.col(21);
	const fvec f_rho = lut.col(22);
	const fvec f_complex = lut.col(23);
	const fvec dir_swdr_lut = lut.col(20);

	//=======par=======
	const fvec f0_par = lut.col(24) + lut.col(25);
	const fvec f_rho_par = lut.col(26);
	const fvec f_complex_par = lut.col(27);
	const fvec dir_par_lut = lut.col(24);

	//=======uva=======
	const fvec f0_uva = lut.col(28) + lut.col(29);
	const fvec f_rho_uva = lut.col(30);
	const fvec f_complex_uva = lut.col(31);
	const fvec dir_uva_lut = lut.col(28);

	//=======uvb=======
	const fvec f0_uvb = lut.col(32) + lut.col(33);
	const fvec f_rho_uvb = lut.col(34);
	const fvec f_complex_uvb = lut.col(35);
	const fvec dir_uvb_lut = lut.col(32);

	//=======toa_albedo(toa_up_flux)=======
	const fvec f0_albedo = lut.col(36);
	const fvec f_rho_albedo = lut.col(37);
	const fvec f_complex_albedo = lut.col(38);
	const fvec f_toa_dw_flux = lut.col(39);

	//--------------------------------------------
	//=======swdr========
	fmat f0_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	f0_tile.each_col() = f0;

	fmat f_rho_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	f_rho_tile.each_col() = f_rho;

	fmat f_complex_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	f_complex_tile.each_col() = f_complex;

	//======par==========
	fmat f0_par_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	f0_par_tile.each_col() = f0_par;

	fmat f_rho_par_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	f_rho_par_tile.each_col() = f_rho_par;

	fmat f_complex_par_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	f_complex_par_tile.each_col() = f_complex_par;

	//======uva==========
	fmat f0_uva_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	f0_uva_tile.each_col() = f0_uva;

	fmat f_rho_uva_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	f_rho_uva_tile.each_col() = f_rho_uva;

	fmat f_complex_uva_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	f_complex_uva_tile.each_col() = f_complex_uva;

	//======uvb==========
	fmat f0_uvb_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	f0_uvb_tile.each_col() = f0_uvb;

	fmat f_rho_uvb_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	f_rho_uvb_tile.each_col() = f_rho_uvb;

	fmat f_complex_uvb_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	f_complex_uvb_tile.each_col() = f_complex_uvb;

	//======toa_albedo==========
	fmat f0_albedo_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	f0_albedo_tile.each_col() = f0_albedo;

	fmat f_rho_albedo_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	f_rho_albedo_tile.each_col() = f_rho_albedo;

	fmat f_complex_albedo_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	f_complex_albedo_tile.each_col() = f_complex_albedo;

	fmat f_toa_dw_flux_tile(nrows_lut_tile, ncols_lut_tile, arma::fill::zeros);
	f_toa_dw_flux_tile.each_col() = f_toa_dw_flux;

	//-----------------------------------------
	//====�ܷ���====
	fmat f0x = trans(sw_albedo_sub_v) % f_rho_tile.each_row();
	fmat swdr_tile = f0_tile + f0x / (1 - f0x) % f_complex_tile;

	//====PAR�ܷ���====
	fmat f0x_par = trans(vis_albedo_sub_v) % f_rho_par_tile.each_row();
	fmat par_tile = f0_par_tile + f0x_par / (1 - f0x_par) % f_complex_par_tile;

	//====UVA�ܷ���====
	fmat f0x_uva = trans(vis_albedo_sub_v) % f_rho_uva_tile.each_row();
	fmat uva_tile = f0_uva_tile + f0x_uva / (1 - f0x_uva) % f_complex_uva_tile;

	//====UVB�ܷ���====
	fmat f0x_uvb = trans(vis_albedo_sub_v) % f_rho_uvb_tile.each_row();
	fmat uvb_tile = f0_uvb_tile + f0x_uvb / (1 - f0x_uvb) % f_complex_uvb_tile;

	//====TOA_albedo====
	fmat f0x_albedo = trans(1 / sw_albedo_sub_v) - f_rho_albedo_tile.each_row();
	fmat toa_albedo_tile = f0_albedo_tile + (1 / f0x_albedo) % f_complex_albedo_tile;
	fmat toa_up_flux_tile = toa_albedo_tile % f_toa_dw_flux_tile; //��Ҫȷ���Ƿ���ÿcol���

	//========================================================================
	fvec ref_mean_sub_v = (band1_ref_sub_v + band3_ref_sub_v) / 2;

	// -------------------------------------------------------
	// (11) flux at (up_sza, up_vza)

	fvec us_uv_swdr;
	fvec us_uv_swdir;
	fvec us_uv_par;
	fvec us_uv_pardir;
	fvec us_uv_uva;
	fvec us_uv_uvb;
	fvec us_uv_toa_up_flux;
	fvec us_uv_rho;

	//-------��ֵus_uv_DEM----------------------------
	int flag = interp_dem(dem_sub_v, toa_rad_b1_sub_v, toa_rad_b3_sub_v, toa_rad_b6_sub_v, toa_rad_b7_sub_v,
		toa_rad_band1_lut_tile, toa_rad_band3_lut_tile, toa_rad_band6_lut_tile, toa_rad_band7_lut_tile,
		toa_rad_band1_lut_clear_tile, toa_rad_band3_lut_clear_tile, toa_rad_band6_lut_clear_tile, toa_rad_band7_lut_clear_tile,
		toa_rad_band1_lut_cloudy_tile, toa_rad_band3_lut_cloudy_tile, toa_rad_band7_lut_cloudy_tile,
		toa_rad_ndsi_lut_tile, lut_diff_max, lut_diff_min,
		idx_us_uv_ud, idx_us_uv_dd,
		swdr_tile, dir_swdr_lut, par_tile, dir_par_lut, uva_tile, uvb_tile, toa_up_flux_tile,
		COD, f_rho, ref_mean_sub_v, band1_ref_sub_v, band3_ref_sub_v, band6_ref_sub_v, band7_ref_sub_v, sza_sub_v,
		us_uv_swdr, us_uv_swdir, us_uv_par, us_uv_pardir, us_uv_uva, us_uv_uvb, us_uv_toa_up_flux, us_uv_rho);
	if (flag != 0) return 1;

	//------------------------------------------------------

	fvec up_sza_swdr = zeros<fvec>(ncols_lut_tile) - 1.0;
	fvec up_sza_swdir = zeros<fvec>(ncols_lut_tile) - 1.0;
	fvec up_sza_par = zeros<fvec>(ncols_lut_tile) - 1.0;
	fvec up_sza_pardir = zeros<fvec>(ncols_lut_tile) - 1.0;
	fvec up_sza_uva = zeros<fvec>(ncols_lut_tile) - 1.0;
	fvec up_sza_uva_dir = zeros<fvec>(ncols_lut_tile) - 1.0;
	fvec up_sza_uvb = zeros<fvec>(ncols_lut_tile) - 1.0;
	fvec up_sza_uvb_dir = zeros<fvec>(ncols_lut_tile) - 1.0;
	fvec up_sza_toa_up_flux = zeros<fvec>(ncols_lut_tile) - 1.0;
	fvec up_sza_rho = zeros<fvec>(ncols_lut_tile) - 1.0;

	if (m_up_vza == m_dw_vza)
	{
		up_sza_swdr = us_uv_swdr;
		up_sza_swdir = us_uv_swdir;
		up_sza_par = us_uv_par;
		up_sza_pardir = us_uv_pardir;
		up_sza_uva = us_uv_uva;
		up_sza_uvb = us_uv_uvb;
		up_sza_toa_up_flux = us_uv_toa_up_flux;
		up_sza_rho = us_uv_rho;
	}
	else
	{
		// -------------------------------------------------------
		// (22) flux at (up_sza, dw_vza)

		fvec us_dv_swdr;
		fvec us_dv_swdir;
		fvec us_dv_par;
		fvec us_dv_pardir;
		fvec us_dv_uva;
		fvec us_dv_uva_dir;
		fvec us_dv_uvb;
		fvec us_dv_uvb_dir;
		fvec us_dv_toa_up_flux;
		fvec us_dv_rho;

		//-------��ֵus_dv_DEM----------------------------

		flag = interp_dem(dem_sub_v, toa_rad_b1_sub_v, toa_rad_b3_sub_v, toa_rad_b6_sub_v, toa_rad_b7_sub_v,
			toa_rad_band1_lut_tile, toa_rad_band3_lut_tile, toa_rad_band6_lut_tile, toa_rad_band7_lut_tile,
			toa_rad_band1_lut_clear_tile, toa_rad_band3_lut_clear_tile, toa_rad_band6_lut_clear_tile, toa_rad_band7_lut_clear_tile,
			toa_rad_band1_lut_cloudy_tile, toa_rad_band3_lut_cloudy_tile, toa_rad_band7_lut_cloudy_tile,
			toa_rad_ndsi_lut_tile, lut_diff_max, lut_diff_min,
			idx_us_dv_ud, idx_us_dv_dd,
			swdr_tile, dir_swdr_lut, par_tile, dir_par_lut, uva_tile, uvb_tile, toa_up_flux_tile,
			COD, f_rho, ref_mean_sub_v, band1_ref_sub_v, band3_ref_sub_v, band6_ref_sub_v, band7_ref_sub_v, sza_sub_v,
			us_dv_swdr, us_dv_swdir, us_dv_par, us_dv_pardir, us_dv_uva, us_dv_uvb, us_dv_toa_up_flux, us_dv_rho);

		if (flag != 0) return 1;

		// -------------------------------------------------------
		// (33) interpolated flux at (up_SZA)

		fmat slope = (us_uv_swdr - us_dv_swdr) / (m_up_vza - m_dw_vza);
		up_sza_swdr = us_dv_swdr + slope % (vza_sub_v - m_dw_vza);

		slope = (us_uv_swdir - us_dv_swdir) / (m_up_vza - m_dw_vza);
		up_sza_swdir = us_dv_swdir + slope % (vza_sub_v - m_dw_vza);

		slope = (us_uv_par - us_dv_par) / (m_up_vza - m_dw_vza);
		up_sza_par = us_dv_par + slope % (vza_sub_v - m_dw_vza);

		slope = (us_uv_pardir - us_dv_pardir) / (m_up_vza - m_dw_vza);
		up_sza_pardir = us_dv_pardir + slope % (vza_sub_v - m_dw_vza);

		slope = (us_uv_uva - us_dv_uva) / (m_up_vza - m_dw_vza);
		up_sza_uva = us_dv_uva + slope % (vza_sub_v - m_dw_vza);

		slope = (us_uv_uvb - us_dv_uvb) / (m_up_vza - m_dw_vza);
		up_sza_uvb = us_dv_uvb + slope % (vza_sub_v - m_dw_vza);

		slope = (us_uv_toa_up_flux - us_dv_toa_up_flux) / (m_up_vza - m_dw_vza);
		up_sza_toa_up_flux = us_dv_toa_up_flux + slope % (vza_sub_v - m_dw_vza);

		slope = (us_uv_rho - us_dv_rho) / (m_up_vza - m_dw_vza);
		up_sza_rho = us_dv_rho + slope % (vza_sub_v - m_dw_vza);

	}

	if (m_up_sza == m_dw_sza)
	{
		derived_swdr = up_sza_swdr;
		derived_dir = up_sza_swdir;
		derived_par = up_sza_par;
		derived_pardir = up_sza_pardir;
		derived_uva = up_sza_uva;
		derived_uvb = up_sza_uvb;
		derived_toa_up_flux = up_sza_toa_up_flux;
		derived_rho = up_sza_rho;
	}

	// -------------------------------------------------------
	// Next interpolated for flux at(dw_SZA) solar(2nd half) 

	// (44) flux at (dw_sza, up_vza)
	fvec ds_uv_swdr;
	fvec ds_uv_swdir;
	fvec ds_uv_par;
	fvec ds_uv_pardir;
	fvec ds_uv_uva;
	fvec ds_uv_uvb;
	fvec ds_uv_toa_up_flux;
	fvec ds_uv_rho;

	//-------��ֵds_uv_DEM----------------------------

	flag = interp_dem(dem_sub_v, toa_rad_b1_sub_v, toa_rad_b3_sub_v, toa_rad_b6_sub_v, toa_rad_b7_sub_v,
		toa_rad_band1_lut_tile, toa_rad_band3_lut_tile, toa_rad_band6_lut_tile, toa_rad_band7_lut_tile,
		toa_rad_band1_lut_clear_tile, toa_rad_band3_lut_clear_tile, toa_rad_band6_lut_clear_tile, toa_rad_band7_lut_clear_tile,
		toa_rad_band1_lut_cloudy_tile, toa_rad_band3_lut_cloudy_tile, toa_rad_band7_lut_cloudy_tile,
		toa_rad_ndsi_lut_tile, lut_diff_max, lut_diff_min,
		idx_ds_uv_ud, idx_ds_uv_dd,
		swdr_tile, dir_swdr_lut, par_tile, dir_par_lut, uva_tile, uvb_tile, toa_up_flux_tile,
		COD, f_rho, ref_mean_sub_v, band1_ref_sub_v, band3_ref_sub_v, band6_ref_sub_v, band7_ref_sub_v, sza_sub_v,
		ds_uv_swdr, ds_uv_swdir, ds_uv_par, ds_uv_pardir, ds_uv_uva, ds_uv_uvb, ds_uv_toa_up_flux, ds_uv_rho);

	if (flag != 0) return 1;

	//------------------------------------------------------

	fvec dw_sza_swdr = zeros<fvec>(ncols_lut_tile) - 1.0;
	fvec dw_sza_swdir = zeros<fvec>(ncols_lut_tile) - 1.0;
	fvec dw_sza_par = zeros<fvec>(ncols_lut_tile) - 1.0;
	fvec dw_sza_pardir = zeros<fvec>(ncols_lut_tile) - 1.0;
	fvec dw_sza_uva = zeros<fvec>(ncols_lut_tile) - 1.0;
	fvec dw_sza_uvb = zeros<fvec>(ncols_lut_tile) - 1.0;
	fvec dw_sza_toa_up_flux = zeros<fvec>(ncols_lut_tile) - 1.0;
	fvec dw_sza_rho = zeros<fvec>(ncols_lut_tile) - 1.0;

	if (m_up_vza == m_dw_vza)
	{
		dw_sza_swdr = ds_uv_swdr;
		dw_sza_swdir = ds_uv_swdir;
		dw_sza_par = ds_uv_par;
		dw_sza_pardir = ds_uv_pardir;
		dw_sza_uva = ds_uv_uva;
		dw_sza_uvb = ds_uv_uvb;
		dw_sza_toa_up_flux = ds_uv_toa_up_flux;
		dw_sza_rho = ds_uv_rho;
	}
	else
	{
		// (55) flux at (dw_sza, dw_vza)
		fvec ds_dv_swdr;
		fvec ds_dv_swdir;
		fvec ds_dv_par;
		fvec ds_dv_pardir;
		fvec ds_dv_uva;
		fvec ds_dv_uvb;
		fvec ds_dv_toa_up_flux;
		fvec ds_dv_rho;

		//-------��ֵds_dv_DEM----------------------------

		flag = interp_dem(dem_sub_v, toa_rad_b1_sub_v, toa_rad_b3_sub_v, toa_rad_b6_sub_v, toa_rad_b7_sub_v,
			toa_rad_band1_lut_tile, toa_rad_band3_lut_tile, toa_rad_band6_lut_tile, toa_rad_band7_lut_tile,
			toa_rad_band1_lut_clear_tile, toa_rad_band3_lut_clear_tile, toa_rad_band6_lut_clear_tile, toa_rad_band7_lut_clear_tile,
			toa_rad_band1_lut_cloudy_tile, toa_rad_band3_lut_cloudy_tile, toa_rad_band7_lut_cloudy_tile,
			toa_rad_ndsi_lut_tile, lut_diff_max, lut_diff_min,
			idx_ds_dv_ud, idx_ds_dv_dd,
			swdr_tile, dir_swdr_lut, par_tile, dir_par_lut, uva_tile, uvb_tile, toa_up_flux_tile,
			COD, f_rho, ref_mean_sub_v, band1_ref_sub_v, band3_ref_sub_v, band6_ref_sub_v, band7_ref_sub_v, sza_sub_v,
			ds_dv_swdr, ds_dv_swdir, ds_dv_par, ds_dv_pardir, ds_dv_uva, ds_dv_uvb, ds_dv_toa_up_flux, ds_dv_rho);

		if (flag != 0) return 1;
		//------------------------------------------------------

		// (66) interpolated flux at (dw_SZA)
		fmat slope = (ds_uv_swdr - ds_dv_swdr) / (m_up_vza - m_dw_vza);
		dw_sza_swdr = ds_dv_swdr + slope % (vza_sub_v - m_dw_vza);

		slope = (ds_uv_swdir - ds_dv_swdir) / (m_up_vza - m_dw_vza);
		dw_sza_swdir = ds_dv_swdir + slope % (vza_sub_v - m_dw_vza);

		slope = (ds_uv_par - ds_dv_par) / (m_up_vza - m_dw_vza);
		dw_sza_par = ds_dv_par + slope % (vza_sub_v - m_dw_vza);

		slope = (ds_uv_pardir - ds_dv_pardir) / (m_up_vza - m_dw_vza);
		dw_sza_pardir = ds_dv_pardir + slope % (vza_sub_v - m_dw_vza);

		slope = (ds_uv_uva - ds_dv_uva) / (m_up_vza - m_dw_vza);
		dw_sza_uva = ds_dv_uva + slope % (vza_sub_v - m_dw_vza);

		slope = (ds_uv_uvb - ds_dv_uvb) / (m_up_vza - m_dw_vza);
		dw_sza_uvb = ds_dv_uvb + slope % (vza_sub_v - m_dw_vza);

		slope = (ds_uv_toa_up_flux - ds_dv_toa_up_flux) / (m_up_vza - m_dw_vza);
		dw_sza_toa_up_flux = ds_dv_toa_up_flux + slope % (vza_sub_v - m_dw_vza);

		slope = (ds_uv_rho - ds_dv_rho) / (m_up_vza - m_dw_vza);
		dw_sza_rho = ds_dv_rho + slope % (vza_sub_v - m_dw_vza);

	}

	// -------------------------------------------------------
	// (77) Final interpolated flux at (SZA)

	fmat slope = (up_sza_swdr - dw_sza_swdr) / (m_up_sza - m_dw_sza);
	fvec itp_swdr = dw_sza_swdr + slope % (sza_sub_v - m_dw_sza);

	slope = (up_sza_swdir - dw_sza_swdir) / (m_up_sza - m_dw_sza);
	fvec itp_swdir = dw_sza_swdir + slope % (sza_sub_v - m_dw_sza);

	slope = (up_sza_par - dw_sza_par) / (m_up_sza - m_dw_sza);
	fvec itp_par = dw_sza_par + slope % (sza_sub_v - m_dw_sza);

	slope = (up_sza_pardir - dw_sza_pardir) / (m_up_sza - m_dw_sza);
	fvec itp_pardir = dw_sza_pardir + slope % (sza_sub_v - m_dw_sza);

	slope = (up_sza_uva - dw_sza_uva) / (m_up_sza - m_dw_sza);
	fvec itp_uva = dw_sza_uva + slope % (sza_sub_v - m_dw_sza);

	slope = (up_sza_uvb - dw_sza_uvb) / (m_up_sza - m_dw_sza);
	fvec itp_uvb = dw_sza_uvb + slope % (sza_sub_v - m_dw_sza);

	slope = (up_sza_toa_up_flux - dw_sza_toa_up_flux) / (m_up_sza - m_dw_sza);
	fvec itp_toa_up_flux = dw_sza_toa_up_flux + slope % (sza_sub_v - m_dw_sza);

	slope = (up_sza_rho - dw_sza_rho) / (m_up_sza - m_dw_sza);
	fvec itp_rho = dw_sza_rho + slope % (sza_sub_v - m_dw_sza);

	// ----------------------------------------------------
	//���ռ�����
	//SWDR
	derived_swdr = itp_swdr;
	derived_dir = itp_swdir;
	derived_par = itp_par;
	derived_pardir = itp_pardir;
	derived_uva = itp_uva;
	derived_uvb = itp_uvb;
	derived_toa_up_flux = itp_toa_up_flux;
	derived_rho = itp_rho;

	return 0;
}


int ahi_swdr::read_lut(const std::string& lut_file, const arma::uword& lut_cols)
{
	using namespace std;
	using namespace arma;

	namespace fs = std::filesystem;

	const size_t ncols = lut_cols;  //LUT������

	if (lut_file.length() == 0)
	{
		cout << "Please specify LUT file\n";
		return 1;
	}

	const fs::path mypath{ lut_file };
	bool flag = fs::exists(mypath);
	if (!flag)
	{
		cout << "[Error] cannot find the LUT file: " << lut_file << endl;
		return 1;
	}

	fs::path lut_path = mypath.parent_path();
	fs::path lut_stem = mypath.stem();
	lut_stem = lut_stem.u8string() + ".bin";
	fs::path lut_bin_file = lut_path / lut_stem;

	if (fs::exists(lut_bin_file))
	{
		cout << "Load LUT directly from " << lut_bin_file.u8string() << endl;
		m_lut.load(lut_bin_file.u8string());
		return 0;
	}

	ifstream fid{ lut_file };
	if (fid.is_open() != 1)
	{
		cout << "Cannot open input file: " << lut_file << endl;
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

	//===============================================================
	//// calculate ncols,�����Զ�������LUT���У����Ǹ���ֵ����
	//string line_head = data[0];
	//vector<string> pack_head;
	//boost::split(pack_head, line_head, boost::is_any_of(" "), boost::token_compress_on);

	//// Remove the invalid element
	//vector<string> pack_head2;
	//for (auto& item : pack_head)
	//{
	//	boost::trim(item);
	//	if (item.length() > 0)
	//		pack_head2.emplace_back(item);
	//}
	//const size_t ncols = pack_head2.size();
	//===============================================================

	// Remove the headers
	data.erase(data.begin());

	// vector to arma::fmat
	const size_t nrows = data.size();
	m_lut = arma::zeros<arma::fmat>(nrows, ncols);

    try {
        for (size_t ir = 0; ir < nrows; ir++)
        {
            const string line = data[ir];

            vector<string> pack;
            boost::split(pack, line, boost::is_any_of(" "), boost::token_compress_on);

            // Remove the invalid element
            vector<string> pack2;
            for (auto& item : pack)
            {
                boost::trim(item);
                if (item.length() > 0)
                    pack2.emplace_back(item);
            }

            if (pack2.size() != ncols)
            {
                cout << "The LUT data is wrong. Only " << pack2.size() << " elements.\n";
                for (const auto& ele : pack2)
                    cout << ele << " ";
                cout << endl;
                return 1;
            }

            for (size_t ic = 0; ic < pack2.size(); ic++)
            {
                tmp = pack2[ic];
                try {
                    double val = std::stod(tmp);
                    m_lut(ir, ic) = static_cast<float>(val);
                } catch (const std::invalid_argument& e) {
                    cerr << "Invalid float format: " << tmp << " in line: " << line << endl;
                    return 1;
                } catch (const std::out_of_range& e) {
                    cerr << "Float value out of range: " << tmp << " in line: " << line << endl;
                    return 1;
                }
            }
        }
    } catch (const std::exception& e) {
        cerr << "Exception occurred: " << e.what() << endl;
        return 1;
    }

	if (!m_lut.save(lut_bin_file.u8string()))
		return 1;
	cout << "Save the LUT file to the binary file: " << lut_bin_file << endl;

	return 0;
}


//������sza, Ҳ������vza
int ahi_swdr::filter_sza(float sza_min, float sza_max, arma::uword& up_sza_idx, arma::uword& dw_sza_idx, const arma::fvec& angle_list, arma::uword m_angle_min, arma::uword m_angle_max)
{
	using namespace std;
	using namespace arma;
	uvec idx;

	// -------------------------------------------------------------------
	//(1)�����ֵ��up_index
	if (sza_max > m_angle_max) // sza > max
	{
		idx = find(angle_list == m_angle_max);
		if (idx.n_elem == 0)
		{
			std::cout << "[Error] cannot find valid elements in sza: " << m_angle_max << endl;
			return 1;
		}

		up_sza_idx = idx(0);
	}
	else
	{
		idx = find(angle_list >= sza_max);
		if (idx.n_elem == 0)
		{
			std::cout << "[Error] cannot find the correct sza: " << sza_max << endl;
			return 1;
		}
		up_sza_idx = idx(0);		
	}

	//(2)����Сֵ��dw_index
	if (sza_min < m_angle_min) // sza < min 
	{

		idx = find(angle_list == m_angle_min);
		if (idx.n_elem == 0)
		{
			std::cout << "[Error] cannot find valid elements in SZA: " << m_angle_min << endl;
			return 1;
		}
		dw_sza_idx = idx(0);
	}
	else
	{
		idx = find(angle_list <= sza_min);
		if (idx.n_elem == 0)
		{
			std::cout << "[Error] cannot find the correct sza: " << sza_min << endl;
			return 1;
		}
		dw_sza_idx = idx(idx.n_elem - 1);		
	}

	return 0;
}


int ahi_swdr::interp_dem(arma::fvec& dem_sub_v, arma::fvec& toa_rad_b1_sub_v, arma::fvec& toa_rad_b3_sub_v, arma::fvec& toa_rad_b6_sub_v, arma::fvec& toa_rad_b7_sub_v,
	arma::fmat toa_rad_band1_lut_tile, arma::fmat toa_rad_band3_lut_tile, arma::fmat toa_rad_band6_lut_tile, arma::fmat toa_rad_band7_lut_tile,
	arma::fmat toa_rad_band1_lut_clear_tile, arma::fmat toa_rad_band3_lut_clear_tile, arma::fmat toa_rad_band6_lut_clear_tile, arma::fmat toa_rad_band7_lut_clear_tile,
	arma::fmat toa_rad_band1_lut_cloudy_tile, arma::fmat& toa_rad_band3_lut_cloudy_tile, arma::fmat toa_rad_band7_lut_cloudy_tile,
	arma::fmat toa_rad_ndsi_lut_tile, float lut_diff_max, float lut_diff_min,
	arma::uword& idx_up_dem, arma::uword& idx_dw_dem,
	arma::fmat swdr_lut_tile, arma::fvec swdr_dir_lut_tile, arma::fmat par_lut_tile, arma::fvec par_dir_lut_tile, arma::fmat uva_lut_tile,
	arma::fmat uvb_lut_tile, arma::fmat toa_up_flux_lut_tile,
	arma::fvec& COD, const arma::fvec& f_rho, arma::fvec& ref_mean_sub_v, arma::fvec& ref_band1_sub_v, arma::fvec& ref_band3_sub_v, arma::fvec& ref_band6_sub_v, arma::fvec& ref_band7_sub_v, arma::fvec& sza_sub_v,
	arma::fvec& itp_swdr, arma::fvec& itp_swdir, arma::fvec& itp_par, arma::fvec& itp_pardir,
	arma::fvec& itp_uva, arma::fvec& itp_uvb, arma::fvec& itp_toa_up_flux, arma::fvec& itp_rho) const
{
	using namespace std;
	using namespace arma;

	int toa_avg_num = m_toa_avg_num;
	const float f_std = m_f_std;

	//=========================================================
	//LUT�ֿ��з���ļ���ֵ
	//// (1) Interpolation - up_DEM
	fmat swdr_lut_dem1_mat = swdr_lut_tile.rows(span(idx_up_dem, idx_up_dem + idx_filter_dem - 1));
	fvec swdr_dir_lut_dem1_v = swdr_dir_lut_tile.rows(span(idx_up_dem, idx_up_dem + idx_filter_dem - 1));
	fmat par_lut_dem1_mat = par_lut_tile.rows(span(idx_up_dem, idx_up_dem + idx_filter_dem - 1));
	fvec par_dir_lut_dem1_v = par_dir_lut_tile.rows(span(idx_up_dem, idx_up_dem + idx_filter_dem - 1));
	fmat uva_lut_dem1_mat = uva_lut_tile.rows(span(idx_up_dem, idx_up_dem + idx_filter_dem - 1));
	fmat uvb_lut_dem1_mat = uvb_lut_tile.rows(span(idx_up_dem, idx_up_dem + idx_filter_dem - 1));
	fmat toa_up_flux_lut_dem1_mat = toa_up_flux_lut_tile.rows(span(idx_up_dem, idx_up_dem + idx_filter_dem - 1));
	fvec rho_lut_dem1_v = f_rho(span(idx_up_dem, idx_up_dem + idx_filter_dem - 1));
	fvec COD_dem1_v = COD(span(idx_up_dem, idx_up_dem + idx_filter_dem - 1));

	// (2) Interpolation - dw_DEM
	fmat swdr_lut_dem2_mat = swdr_lut_tile.rows(span(idx_dw_dem, idx_dw_dem + idx_filter_dem - 1));
	fvec swdr_dir_lut_dem2_v = swdr_dir_lut_tile.rows(span(idx_dw_dem, idx_dw_dem + idx_filter_dem - 1));
	fmat par_lut_dem2_mat = par_lut_tile.rows(span(idx_dw_dem, idx_dw_dem + idx_filter_dem - 1));
	fvec par_dir_lut_dem2_v = par_dir_lut_tile.rows(span(idx_dw_dem, idx_dw_dem + idx_filter_dem - 1));
	fmat uva_lut_dem2_mat = uva_lut_tile.rows(span(idx_dw_dem, idx_dw_dem + idx_filter_dem - 1));
	fmat uvb_lut_dem2_mat = uvb_lut_tile.rows(span(idx_dw_dem, idx_dw_dem + idx_filter_dem - 1));
	fmat toa_up_flux_lut_dem2_mat = toa_up_flux_lut_tile.rows(span(idx_dw_dem, idx_dw_dem + idx_filter_dem - 1));
	fvec rho_lut_dem2_v = f_rho(span(idx_dw_dem, idx_dw_dem + idx_filter_dem - 1));
	fvec COD_dem2_v = COD(span(idx_dw_dem, idx_dw_dem + idx_filter_dem - 1));

	//��ѩ�ʹ����ֿ�=======================================
	//dem1-------------------------------------------------
	//��ѩtoa_rad
	fmat toa_rad_ndsi_dem1_mat = toa_rad_ndsi_lut_tile.rows(span(idx_up_dem, idx_up_dem + idx_filter_dem - 1));

	//���toa_rad
	fmat toa_rad_band1_clear_dem1_mat = toa_rad_band1_lut_clear_tile.rows(span(idx_up_dem, idx_up_dem + idx_filter_dem - 1));//��ʵһ��idx_up_dem��Ӧ��Ӧ��ȫ��һ��ֵ
	fmat toa_rad_band3_clear_dem1_mat = toa_rad_band3_lut_clear_tile.rows(span(idx_up_dem, idx_up_dem + idx_filter_dem - 1));
	fmat toa_rad_band6_clear_dem1_mat = toa_rad_band6_lut_clear_tile.rows(span(idx_up_dem, idx_up_dem + idx_filter_dem - 1));
	fmat toa_rad_band7_clear_dem1_mat = toa_rad_band7_lut_clear_tile.rows(span(idx_up_dem, idx_up_dem + idx_filter_dem - 1));

	//����toa_rad
	fmat toa_rad_band1_cloudy_dem1_mat = toa_rad_band1_lut_cloudy_tile.rows(span(idx_up_dem, idx_up_dem + idx_filter_dem - 1));
	fmat toa_rad_band3_cloudy_dem1_mat = toa_rad_band3_lut_cloudy_tile.rows(span(idx_up_dem, idx_up_dem + idx_filter_dem - 1));
	fmat toa_rad_band7_cloudy_dem1_mat = toa_rad_band7_lut_cloudy_tile.rows(span(idx_up_dem, idx_up_dem + idx_filter_dem - 1));

	//lutģ���toa_rad
	fmat finded_toa_rad_band1_dem1_mat = toa_rad_band1_lut_tile.rows(span(idx_up_dem, idx_up_dem + idx_filter_dem - 1));
	fmat finded_toa_rad_band3_dem1_mat = toa_rad_band3_lut_tile.rows(span(idx_up_dem, idx_up_dem + idx_filter_dem - 1));
	fmat finded_toa_rad_band6_dem1_mat = toa_rad_band6_lut_tile.rows(span(idx_up_dem, idx_up_dem + idx_filter_dem - 1));
	fmat finded_toa_rad_band7_dem1_mat = toa_rad_band7_lut_tile.rows(span(idx_up_dem, idx_up_dem + idx_filter_dem - 1));

	//dem2-------------------------------------------------
	//
	fmat toa_rad_ndsi_dem2_mat = toa_rad_ndsi_lut_tile.rows(span(idx_dw_dem, idx_dw_dem + idx_filter_dem - 1));
	//
	fmat toa_rad_band1_clear_dem2_mat = toa_rad_band1_lut_clear_tile.rows(span(idx_dw_dem, idx_dw_dem + idx_filter_dem - 1));
	fmat toa_rad_band3_clear_dem2_mat = toa_rad_band3_lut_clear_tile.rows(span(idx_dw_dem, idx_dw_dem + idx_filter_dem - 1));
	fmat toa_rad_band6_clear_dem2_mat = toa_rad_band6_lut_clear_tile.rows(span(idx_dw_dem, idx_dw_dem + idx_filter_dem - 1));
	fmat toa_rad_band7_clear_dem2_mat = toa_rad_band7_lut_clear_tile.rows(span(idx_dw_dem, idx_dw_dem + idx_filter_dem - 1));
	//
	fmat toa_rad_band1_cloudy_dem2_mat = toa_rad_band1_lut_cloudy_tile.rows(span(idx_dw_dem, idx_dw_dem + idx_filter_dem - 1));
	fmat toa_rad_band3_cloudy_dem2_mat = toa_rad_band3_lut_cloudy_tile.rows(span(idx_dw_dem, idx_dw_dem + idx_filter_dem - 1));
	fmat toa_rad_band7_cloudy_dem2_mat = toa_rad_band7_lut_cloudy_tile.rows(span(idx_dw_dem, idx_dw_dem + idx_filter_dem - 1));
	//
	fmat finded_toa_rad_band1_dem2_mat = toa_rad_band1_lut_tile.rows(span(idx_dw_dem, idx_dw_dem + idx_filter_dem - 1));
	fmat finded_toa_rad_band3_dem2_mat = toa_rad_band3_lut_tile.rows(span(idx_dw_dem, idx_dw_dem + idx_filter_dem - 1));
	fmat finded_toa_rad_band6_dem2_mat = toa_rad_band6_lut_tile.rows(span(idx_dw_dem, idx_dw_dem + idx_filter_dem - 1));
	fmat finded_toa_rad_band7_dem2_mat = toa_rad_band7_lut_tile.rows(span(idx_dw_dem, idx_dw_dem + idx_filter_dem - 1));

////------------------------------------------------------------
	//����Ľ��
	fvec finded_swdr_sub_v_dem1(toa_rad_b3_sub_v.n_rows, arma::fill::zeros);
	fvec finded_swdr_dir_sub_v_dem1(toa_rad_b3_sub_v.n_rows, arma::fill::zeros);
	fvec finded_par_sub_v_dem1(toa_rad_b3_sub_v.n_rows, arma::fill::zeros);
	fvec finded_par_dir_sub_v_dem1(toa_rad_b3_sub_v.n_rows, arma::fill::zeros);
	fvec finded_uva_sub_v_dem1(toa_rad_b3_sub_v.n_rows, arma::fill::zeros);
	fvec finded_uvb_sub_v_dem1(toa_rad_b3_sub_v.n_rows, arma::fill::zeros);
	fvec finded_toa_up_flux_sub_v_dem1(toa_rad_b3_sub_v.n_rows, arma::fill::zeros);
	fvec finded_rho_sub_v_dem1(toa_rad_b3_sub_v.n_rows, arma::fill::zeros);

	fvec finded_swdr_sub_v_dem2(toa_rad_b3_sub_v.n_rows, arma::fill::zeros);
	fvec finded_swdr_dir_sub_v_dem2(toa_rad_b3_sub_v.n_rows, arma::fill::zeros);
	fvec finded_par_sub_v_dem2(toa_rad_b3_sub_v.n_rows, arma::fill::zeros);
	fvec finded_par_dir_sub_v_dem2(toa_rad_b3_sub_v.n_rows, arma::fill::zeros);
	fvec finded_uva_sub_v_dem2(toa_rad_b3_sub_v.n_rows, arma::fill::zeros);
	fvec finded_uvb_sub_v_dem2(toa_rad_b3_sub_v.n_rows, arma::fill::zeros);
	fvec finded_toa_up_flux_sub_v_dem2(toa_rad_b3_sub_v.n_rows, arma::fill::zeros);
	fvec finded_rho_sub_v_dem2(toa_rad_b3_sub_v.n_rows, arma::fill::zeros);

	////===================================================================================================================================
	////========================================================================================================
	////���ԸĶ�ȡLUT�ķ�ʽ��SZAһ�飬VZA��ȡSZA�ģ���������ؼ���
	for (uword i = 0; i < toa_rad_b3_sub_v.n_rows; i++)
	{
		int toa_avg_num = m_toa_avg_num;

		//-------------------------------------------------------
		//�۲�ֵ
		float toa_rad_b1 = toa_rad_b1_sub_v(i);
		float toa_rad_b3 = toa_rad_b3_sub_v(i);
		float toa_rad_b6 = toa_rad_b6_sub_v(i);
		float toa_rad_b7 = toa_rad_b7_sub_v(i);

		float ref_mean = ref_mean_sub_v(i);
		float ref_band1 = ref_band1_sub_v(i);
		float ref_band3 = ref_band3_sub_v(i);
		float ref_band6 = ref_band6_sub_v(i);
		float ref_band7 = ref_band7_sub_v(i);
		float sza = sza_sub_v(i);

		//////-----����ÿ�����ε�toa_radiance---------
		//-----dem1---------------------------
		//��ѩtoa_rad
		fvec toa_rad_ndsi_dem1 = toa_rad_ndsi_dem1_mat.col(i);

		//���toa_rad
		float toa_rad_b1_clear_dem1 = mean(toa_rad_band1_clear_dem1_mat.col(i));
		float toa_rad_b3_clear_dem1 = mean(toa_rad_band3_clear_dem1_mat.col(i));
		float toa_rad_b6_clear_dem1 = mean(toa_rad_band6_clear_dem1_mat.col(i));
		float toa_rad_b7_clear_dem1 = mean(toa_rad_band7_clear_dem1_mat.col(i));

		//����toa_rad
		float toa_rad_b1_cloudy_dem1 = mean(toa_rad_band1_cloudy_dem1_mat.col(i));
		float toa_rad_b3_cloudy_dem1 = mean(toa_rad_band3_cloudy_dem1_mat.col(i));
		float toa_rad_b7_cloudy_dem1 = mean(toa_rad_band7_cloudy_dem1_mat.col(i));

		//lutģ���toa_rad
		fvec finded_toa_rad_band1_dem1 = finded_toa_rad_band1_dem1_mat.col(i);
		fvec finded_toa_rad_band3_dem1 = finded_toa_rad_band3_dem1_mat.col(i);
		fvec finded_toa_rad_band6_dem1 = finded_toa_rad_band6_dem1_mat.col(i);
		fvec finded_toa_rad_band7_dem1 = finded_toa_rad_band7_dem1_mat.col(i);

		////�ܷ���
		fvec swdr_lut_dem1 = swdr_lut_dem1_mat.col(i);
		fvec par_lut_dem1 = par_lut_dem1_mat.col(i);
		fvec uva_lut_dem1 = uva_lut_dem1_mat.col(i);
		fvec uvb_lut_dem1 = uvb_lut_dem1_mat.col(i);
		fvec toa_up_flux_lut_dem1 = toa_up_flux_lut_dem1_mat.col(i);

		//-----dem2---------------------------
		//��ѩtoa_rad
		fvec toa_rad_ndsi_dem2 = toa_rad_ndsi_dem2_mat.col(i);

		//���toa_rad
		float toa_rad_b1_clear_dem2 = mean(toa_rad_band1_clear_dem2_mat.col(i));
		float toa_rad_b3_clear_dem2 = mean(toa_rad_band3_clear_dem2_mat.col(i));
		float toa_rad_b6_clear_dem2 = mean(toa_rad_band6_clear_dem2_mat.col(i));
		float toa_rad_b7_clear_dem2 = mean(toa_rad_band7_clear_dem2_mat.col(i));

		//����toa_rad
		float toa_rad_b1_cloudy_dem2 = mean(toa_rad_band1_cloudy_dem2_mat.col(i));
		float toa_rad_b3_cloudy_dem2 = mean(toa_rad_band3_cloudy_dem2_mat.col(i));
		float toa_rad_b7_cloudy_dem2 = mean(toa_rad_band7_cloudy_dem2_mat.col(i));

		//lutģ���toa_rad
		fvec finded_toa_rad_band1_dem2 = finded_toa_rad_band1_dem2_mat.col(i);
		fvec finded_toa_rad_band3_dem2 = finded_toa_rad_band3_dem2_mat.col(i);
		fvec finded_toa_rad_band6_dem2 = finded_toa_rad_band6_dem2_mat.col(i);
		fvec finded_toa_rad_band7_dem2 = finded_toa_rad_band7_dem2_mat.col(i);
		
		////�ܷ���
		fvec swdr_lut_dem2 = swdr_lut_dem2_mat.col(i);
		fvec par_lut_dem2 = par_lut_dem2_mat.col(i);
		fvec uva_lut_dem2 = uva_lut_dem2_mat.col(i);
		fvec uvb_lut_dem2 = uvb_lut_dem2_mat.col(i);
		fvec toa_up_flux_lut_dem2 = toa_up_flux_lut_dem2_mat.col(i);

		//========================================================================
		//���ݻ�ѩָ���ֿ�
		float vstd_ndsi = stddev(toa_rad_ndsi_lut_tile.col(i));

		uvec idx_snow_dem1 = find(toa_rad_ndsi_dem1 <= (lut_diff_max + 5 * vstd_ndsi) && toa_rad_ndsi_dem1 >= (lut_diff_min - 5 * vstd_ndsi)); //��ѩ�ֿ����
		//
		if (idx_snow_dem1.n_elem == 0)
		{
			idx_snow_dem1 = find(toa_rad_ndsi_dem1 <= 1 && toa_rad_ndsi_dem1 >= -1);
		}
		//
		//�ж�����toa_avg_numֵ
		if (idx_snow_dem1.n_elem < toa_avg_num)
		{
			toa_avg_num = idx_snow_dem1.n_elem;
		}
		//
		//lutģ���toa_rad
		finded_toa_rad_band1_dem1 = finded_toa_rad_band1_dem1(idx_snow_dem1);
		finded_toa_rad_band3_dem1 = finded_toa_rad_band3_dem1(idx_snow_dem1);
		finded_toa_rad_band6_dem1 = finded_toa_rad_band6_dem1(idx_snow_dem1);
		finded_toa_rad_band7_dem1 = finded_toa_rad_band7_dem1(idx_snow_dem1);
		
		swdr_lut_dem1 = swdr_lut_dem1(idx_snow_dem1);
		fvec swdr_dir_lut_dem1 = swdr_dir_lut_dem1_v(idx_snow_dem1);
		par_lut_dem1 = par_lut_dem1(idx_snow_dem1);
		fvec par_dir_lut_dem1 = par_dir_lut_dem1_v(idx_snow_dem1);
		uva_lut_dem1 = uva_lut_dem1(idx_snow_dem1);
		uvb_lut_dem1 = uvb_lut_dem1(idx_snow_dem1);
		toa_up_flux_lut_dem1 = toa_up_flux_lut_dem1(idx_snow_dem1);
		fvec rho_lut_dem1 = rho_lut_dem1_v(idx_snow_dem1);
		fvec COD_dem1 = COD_dem1_v(idx_snow_dem1);

		//-------------------------------------------------------------------
		//float vstd2 = stddev(toa_rad_ndsi_dem2);
		uvec idx_snow_dem2 = find(toa_rad_ndsi_dem2 <= (lut_diff_max + 5 * vstd_ndsi) && toa_rad_ndsi_dem2 >= (lut_diff_min - 5 * vstd_ndsi));
		//
		if (idx_snow_dem2.n_elem == 0)
		{
			idx_snow_dem2 = find(toa_rad_ndsi_dem2 <= 1 && toa_rad_ndsi_dem2 >= -1);
		}
		//
		//�ж�����toa_avg_numֵ
		if (idx_snow_dem2.n_elem < toa_avg_num)
		{
			toa_avg_num = idx_snow_dem2.n_elem;
		}
		//
		//lutģ���toa_rad
		finded_toa_rad_band1_dem2 = finded_toa_rad_band1_dem2(idx_snow_dem2);
		finded_toa_rad_band3_dem2 = finded_toa_rad_band3_dem2(idx_snow_dem2);
		finded_toa_rad_band6_dem2 = finded_toa_rad_band6_dem2(idx_snow_dem2);
		finded_toa_rad_band7_dem2 = finded_toa_rad_band7_dem2(idx_snow_dem2);
		
		swdr_lut_dem2 = swdr_lut_dem2(idx_snow_dem2);
		fvec swdr_dir_lut_dem2 = swdr_dir_lut_dem2_v(idx_snow_dem2);
		par_lut_dem2 = par_lut_dem2(idx_snow_dem2);
		fvec par_dir_lut_dem2 = par_dir_lut_dem2_v(idx_snow_dem2);
		uva_lut_dem2 = uva_lut_dem2(idx_snow_dem2);
		uvb_lut_dem2 = uvb_lut_dem2(idx_snow_dem2);
		toa_up_flux_lut_dem2 = toa_up_flux_lut_dem2(idx_snow_dem2);
		fvec rho_lut_dem2 = rho_lut_dem2_v(idx_snow_dem2);
		fvec COD_dem2 = COD_dem2_v(idx_snow_dem2);

		//========================================================================
		//�۲�ֵtoa_rad
		fvec toa_rad = { toa_rad_b1, toa_rad_b3, toa_rad_b7 };
		fvec toa_rad_cloud = { toa_rad_b3, toa_rad_b6, toa_rad_b7 };

		//ģ��ֵtoa_rad
		fmat toa_rad_lut_dem1 = zeros<fmat>(finded_toa_rad_band3_dem1.n_rows, 3);
		toa_rad_lut_dem1.col(0) = finded_toa_rad_band1_dem1;
		toa_rad_lut_dem1.col(1) = finded_toa_rad_band3_dem1;
		toa_rad_lut_dem1.col(2) = finded_toa_rad_band7_dem1;

		fmat toa_rad_lut_dem2 = zeros<fmat>(finded_toa_rad_band3_dem2.n_rows, 3);
		toa_rad_lut_dem2.col(0) = finded_toa_rad_band1_dem2;
		toa_rad_lut_dem2.col(1) = finded_toa_rad_band3_dem2;
		toa_rad_lut_dem2.col(2) = finded_toa_rad_band7_dem2;

		//ģ��ֵtoa_rad,����
		fmat toa_rad_lut_dem1_cloud = zeros<fmat>(finded_toa_rad_band3_dem1.n_rows, 3);
		toa_rad_lut_dem1_cloud.col(0) = finded_toa_rad_band3_dem1;
		toa_rad_lut_dem1_cloud.col(1) = finded_toa_rad_band6_dem1;
		toa_rad_lut_dem1_cloud.col(2) = finded_toa_rad_band7_dem1;

		fmat toa_rad_lut_dem2_cloud = zeros<fmat>(finded_toa_rad_band3_dem2.n_rows, 3);
		toa_rad_lut_dem2_cloud.col(0) = finded_toa_rad_band3_dem2;
		toa_rad_lut_dem2_cloud.col(1) = finded_toa_rad_band6_dem2;
		toa_rad_lut_dem2_cloud.col(2) = finded_toa_rad_band7_dem2;
		//==============================================================

		//�жϴ�������
		//0-��ȷ����1-��գ�2-60���±��ƣ�3-����
		//dem1
		uword clear_flag_dem1 = 0;//���
		//���
		float toa_rad_change_dem1 = (abs(toa_rad_b1 - toa_rad_b1_clear_dem1) / toa_rad_b1_clear_dem1 + abs(toa_rad_b3 - toa_rad_b3_clear_dem1) / toa_rad_b3_clear_dem1
			+ abs(toa_rad_b7 - toa_rad_b7_clear_dem1) / toa_rad_b7_clear_dem1) / 3;
		if (toa_rad_change_dem1 >= 0.2)
		{
			clear_flag_dem1 = 2; //����Ϊ����
		}

		//�������==================================
		//����������������ı��Ϊ����
		if (ref_mean < 0.60 && (toa_rad_b1 > toa_rad_b1_cloudy_dem1*0.95) && (toa_rad_b3 > toa_rad_b3_cloudy_dem1*0.95) && (toa_rad_b7 < toa_rad_b7_cloudy_dem1))
		{

			clear_flag_dem1 = 3;
		}

		//======================================================================
		//----------------------------------------------------------------------
		//dem2,�����ж���ն���=================================================
		uword clear_flag_dem2 = 0;//flagΪ1���Ϊ���
		//���
		float toa_rad_change_dem2 = (abs(toa_rad_b1 - toa_rad_b1_clear_dem2) / toa_rad_b1_clear_dem2 + abs(toa_rad_b3 - toa_rad_b3_clear_dem2) / toa_rad_b3_clear_dem2
			+ abs(toa_rad_b7 - toa_rad_b7_clear_dem2) / toa_rad_b7_clear_dem2) / 3;
		if (toa_rad_change_dem2 >= 0.2)
		{
			clear_flag_dem2 = 2; //����Ϊ����
		}

		//�������======================================================
		//����������������ı��Ϊ����
		if (ref_mean < 0.60 && (toa_rad_b1 > toa_rad_b1_cloudy_dem2*0.95) && (toa_rad_b3 > toa_rad_b3_cloudy_dem2*0.95) && (toa_rad_b7 < toa_rad_b7_cloudy_dem2))
		{
			clear_flag_dem2 = 3;
		}
		
		//========================================================================================================
		//cout << clear_flag_dem1 << endl;
		//cout << clear_flag_dem2 << endl;
		////-----ԭʼ�����Σ��þ���ֵ----------------------------------
		fvec tmpv_dem1 = abs(finded_toa_rad_band3_dem1 - toa_rad_b3); //dw_dem(dem1)
		fvec tmpv_dem2 = abs(finded_toa_rad_band3_dem2 - toa_rad_b3); //up_dem(dem2)
		if (tmpv_dem1.has_nan() || tmpv_dem1.has_nan())
		{
			cout << "tmpv_dem1: " << tmpv_dem1 << endl;
			cout << "tmpv_dem2: " << tmpv_dem2 << endl;
		}
		uvec idx2 = sort_index(tmpv_dem1);
		idx2 = idx2.head(toa_avg_num);

		//==========================�ನ���㷨====================================================================
		uvec idx_total; 
		uvec idx_dir; 
		//-----------------------------------------------
		//�ֲ�ͬ�Ĵ����������idx_total��idx_dir
		//-----------------------------------------------
		if (clear_flag_dem1 == 1) //���
		{
			fmat toa_rad_lut_dem1_log = log(toa_rad_lut_dem1);
			fmat tmpv_dem1_mat = pow((toa_rad_lut_dem1_log.each_row() - trans(log(toa_rad))), 2);

			tmpv_dem1 = sqrt(sum(tmpv_dem1_mat, 1));
			if (tmpv_dem1.has_nan())
			{
				cout << "tmpv_dem1: " << tmpv_dem1 << endl;
			}

			uvec idx1 = sort_index(tmpv_dem1);
			idx1 = idx1.head(toa_avg_num);

			//�ж�idx1��1����
			fvec COD_dem1_revise = COD_dem1.rows(idx1);
			uvec idx1_revise;
			idx1_revise = find(COD_dem1_revise <= 1);

			int toa_avg_num_new = toa_avg_num;
			while (idx1_revise.n_elem < 2)
			{
				idx1 = sort_index(tmpv_dem1);
				toa_avg_num_new = toa_avg_num_new + 1;
				if (toa_avg_num_new > idx1.n_elem)
				{
					idx1_revise = find(COD_dem1_revise >= 0); //�൱�ڲ���������������idx1����
					break;
				}

				idx1 = idx1.head(toa_avg_num_new);
				COD_dem1_revise = COD_dem1.rows(idx1);
				idx1_revise = find(COD_dem1_revise <= 1);
			}

			idx1 = idx1.rows(idx1_revise);

			//idx1����������blue band������
			uvec idx3 = intersect(idx1, idx2);
			if (idx3.n_elem > 4)
			{
				idx2 = idx3;
			}
			else
			{
				idx2 = idx1;
			}

			//idx_total = idx2; //ln+ŷʽ,blue,�������ܷ�������Ⲩ��ȥ����
			idx_total = idx1;
			idx_dir = idx1; //ln+ŷʽ

		}
		else if (clear_flag_dem1 == 3)//���ƣ�
		{
			//��״��cos
			//��������ֵ
			float toa_revise0 = mean(toa_rad_cloud);  //�۲�ֵ��ֵ	

			//����
			fvec cosine_multiband_dem1(toa_rad_lut_dem1_cloud.n_rows, arma::fill::zeros);
			fmat toa_rad_lut_dem1_cloud_log = log(toa_rad_lut_dem1_cloud);
			for (uword i = 0; i < toa_rad_lut_dem1_cloud.n_rows; i++)
			{

				cosine_multiband_dem1(i) = norm_dot(toa_rad_lut_dem1_cloud_log.row(i) - log(toa_revise0), trans(log(toa_rad_cloud)) - log(toa_revise0));
			}

			uvec idx1 = sort_index(cosine_multiband_dem1, "descend");
			idx1 = idx1.head(toa_avg_num);

			idx_total = idx1;
			idx_dir = idx1;
		}
		else//���ƣ�����,��ȷ��
		{
			//��������ֵ
			float toa_revise0 = mean(toa_rad);  //�۲�ֵ��ֵ	

			//����
			fvec cosine_multiband_dem1(toa_rad_lut_dem1.n_rows, arma::fill::zeros);
			fmat toa_rad_lut_dem1_log = log(toa_rad_lut_dem1);

			for (uword i = 0; i < toa_rad_lut_dem1.n_rows; i++)
			{

				cosine_multiband_dem1(i) = norm_dot(toa_rad_lut_dem1_log.row(i) - log(toa_revise0), trans(log(toa_rad)) - log(toa_revise0));

			}
			uvec idx1 = sort_index(cosine_multiband_dem1, "descend");
			idx1 = idx1.head(toa_avg_num);

			//====================================
			if ((ref_band3 < 0.82)&(ref_band3 >= 0.65))
			{
				//��������ֵ
				toa_revise0 = mean(toa_rad_cloud);  //�۲�ֵ��ֵ	

				//����
				fvec cosine_multiband_dem1_2(toa_rad_lut_dem1_cloud.n_rows, arma::fill::zeros);
				fmat toa_rad_lut_dem1_log_2 = log(toa_rad_lut_dem1_cloud);

				for (uword i = 0; i < toa_rad_lut_dem1_cloud.n_rows; i++)
				{

					cosine_multiband_dem1_2(i) = norm_dot(toa_rad_lut_dem1_log_2.row(i) - log(toa_revise0), trans(log(toa_rad_cloud)) - log(toa_revise0));

				}
				idx1 = sort_index(cosine_multiband_dem1_2, "descend");
				idx1 = idx1.head(toa_avg_num);
			}
			//====================================

			//�ж�idx1��60����
			fvec COD_dem1_revise = COD_dem1.rows(idx1);
			uvec idx1_revise;
			idx1_revise = find(COD_dem1_revise <= 60 && COD_dem1_revise >= 0);

			int toa_avg_num_new = toa_avg_num;
			while (idx1_revise.n_elem < 2)
			{
				idx1 = sort_index(cosine_multiband_dem1, "descend");
				toa_avg_num_new = toa_avg_num_new + 1;
				if (toa_avg_num_new > idx1.n_elem)
				{
					idx1_revise = find(COD_dem1_revise >= 0); //�൱�ڲ���������������idx1����
					break;
				}

				idx1 = idx1.head(toa_avg_num_new);

				COD_dem1_revise = COD_dem1.rows(idx1);
				idx1_revise = find(COD_dem1_revise <= 60 && COD_dem1_revise >= 0);
			}

			idx1 = idx1.rows(idx1_revise);

			//�������ཻ
			uvec idx3 = intersect(idx1, idx2);
			if (idx3.n_elem > 4)
			{
				idx2 = idx3;
			}

			COD_dem1_revise = COD_dem1.rows(idx2);
			uvec idx2_revise = find(COD_dem1_revise <= 60 && COD_dem1_revise >= 0);

			if (idx2_revise.n_elem != 0)
			{
				idx2 = idx2.rows(idx2_revise);
			}
			else
			{
				idx2 = idx1;
			}

			//idx_total = idx2; //ln+cos,blue,����
			idx_total = idx1;
			idx_dir = idx1; //ln+cos
		}

		//----swdr-dem1----;
		if ((ref_mean < 0.3) && (clear_flag_dem1 != 3))
		{
			idx_total = idx2;
		}
		fvec swdr_dem1 = swdr_lut_dem1(idx_total);		
		fvec rho_dem1 = rho_lut_dem1(idx_total);
		//
		float vmean = mean(swdr_dem1);
		float vstd = stddev(swdr_dem1) * f_std;
		uvec idx = find(swdr_dem1 <= (vmean + vstd + 0.1) && swdr_dem1 >= (vmean - vstd - 0.1));
		if (idx.n_elem == 0)
		{
			cout << "(1) Interpolation - up_DEM\n";
			cout << "[Error] cannot find valid elements for SWDR range from "
				<< vmean - vstd << " to " << vmean + vstd << endl;
			return 1;
		}

		float swdr_avg1 = mean(swdr_dem1(idx));
		float rho_avg1 = mean(rho_dem1(idx));
		finded_swdr_sub_v_dem1(i) = swdr_avg1;
		finded_rho_sub_v_dem1(i) = rho_avg1;

		//------------------------------------
		//---dir_dem1----
		swdr_dem1 = swdr_lut_dem1(idx_dir);
		fvec dir_dem1 = swdr_dir_lut_dem1(idx_dir);

		vmean = mean(swdr_dem1);
		vstd = stddev(swdr_dem1) * f_std;

		idx = find(swdr_dem1 <= (vmean + vstd + 0.1) && swdr_dem1 >= (vmean - vstd - 0.1));
		if (idx.n_elem == 0)
		{
			std::cout << "[Error_dem1] cannot find valid elements for SWDR(dir) range from "
				<< vmean - vstd << " to " << vmean + vstd << std::endl;
			return 1;
		}
		float dir_avg1 = mean(dir_dem1(idx));
		finded_swdr_dir_sub_v_dem1(i) = dir_avg1;

		//------------------------------------
		//---par_dem1----
		fvec par_dem1 = par_lut_dem1(idx_total);

		vmean = mean(par_dem1);
		vstd = stddev(par_dem1) * f_std;

		idx = find(par_dem1 <= (vmean + vstd + 0.1) && par_dem1 >= (vmean - vstd - 0.1));
		if (idx.n_elem == 0)
		{
			std::cout << "[Error_dem1] cannot find valid elements for PAR range from "
				<< vmean - vstd << " to " << vmean + vstd << std::endl;
			return 1;
		}
		float par_avg1 = mean(par_dem1(idx));
		finded_par_sub_v_dem1(i) = par_avg1;

		//------------------------------------
		//---par_dir_dem1----
		par_dem1 = par_lut_dem1(idx_dir);
		fvec par_dir_dem1 = par_dir_lut_dem1(idx_dir);

		vmean = mean(par_dem1);
		vstd = stddev(par_dem1) * f_std;

		idx = find(par_dem1 <= (vmean + vstd + 0.1) && par_dem1 >= (vmean - vstd - 0.1));
		if (idx.n_elem == 0)
		{
			std::cout << "[Error_dem1] cannot find valid elements for PAR(dir) range from "
				<< vmean - vstd << " to " << vmean + vstd << std::endl;
			return 1;
		}
		float par_dir_avg1 = mean(par_dir_dem1(idx));
		finded_par_dir_sub_v_dem1(i) = par_dir_avg1;

		//------------------------------------
		//-uva_dem1--
		fvec uva_dem1 = uva_lut_dem1(idx_total);

		vmean = mean(uva_dem1);
		vstd = stddev(uva_dem1) * f_std;
		idx = find(uva_dem1 <= (vmean + vstd + 0.01) && uva_dem1 >= (vmean - vstd - 0.01));
		if (idx.n_elem == 0)
		{
			cout << "[Error_dem1] cannot find valid elements for UVA range from "
				<< vmean - vstd << " to " << vmean + vstd << endl;
			return 1;
		}
		float uva_avg1 = mean(uva_dem1(idx));
		finded_uva_sub_v_dem1(i) = uva_avg1;


		//------------------------------------
		//-uvb_dem1--
		fvec uvb_dem1 = uvb_lut_dem1(idx_total);

		vmean = mean(uvb_dem1);
		vstd = stddev(uvb_dem1) * f_std;
		idx = find(uvb_dem1 <= (vmean + vstd + 0.01) && uvb_dem1 >= (vmean - vstd - 0.01));
		if (idx.n_elem == 0)
		{
			cout << "[Error_dem1] cannot find valid elements for UVB range from "
				<< vmean - vstd << " to " << vmean + vstd << endl;
			return 1;
		}
		float uvb_avg1 = mean(uvb_dem1(idx));
		finded_uvb_sub_v_dem1(i) = uvb_avg1;


		//------------------------------------
		//-toa_albedo_dem1--
		fvec toa_up_flux_dem1 = toa_up_flux_lut_dem1(idx_total);

		vmean = mean(toa_up_flux_dem1);
		vstd = stddev(toa_up_flux_dem1) * f_std;
		idx = find(toa_up_flux_dem1 <= (vmean + vstd + 0.1) && toa_up_flux_dem1 >= (vmean - vstd - 0.1));
		if (idx.n_elem == 0)
		{
			cout << "[Error_dem1] cannot find valid elements for TOA_up_flux range from "
				<< vmean - vstd << " to " << vmean + vstd << endl;
			return 1;
		}
		float toa_albedo_avg1 = mean(toa_up_flux_dem1(idx));
		finded_toa_up_flux_sub_v_dem1(i) = toa_albedo_avg1;

		//===================================================
		//---------dem2���ۺ���------------------------
		//========================================================================================================
		////-----ԭʼ�����Σ��þ���ֵ----------------------------------
		idx2 = sort_index(tmpv_dem2);
		idx2 = idx2.head(toa_avg_num);

		//========================================================================================================
		//��������ģ����cos��ŷʽ����Ķನ���㷨
		if (clear_flag_dem2 == 1) //���
		{
			fmat toa_rad_lut_dem2_log = log(toa_rad_lut_dem2);
			fmat tmpv_dem2_mat = pow((toa_rad_lut_dem2_log.each_row() - trans(log(toa_rad))), 2);

			tmpv_dem2 = sqrt(sum(tmpv_dem2_mat, 1));

			uvec idx1 = sort_index(tmpv_dem2);
			idx1 = idx1.head(toa_avg_num);

			//�ж�idx1��1����
			fvec COD_dem2_revise = COD_dem2.rows(idx1);
			uvec idx1_revise;
			idx1_revise = find(COD_dem2_revise <= 1);

			int toa_avg_num_new = toa_avg_num;
			while (idx1_revise.n_elem < 2)
			{
				idx1 = sort_index(tmpv_dem2);
				toa_avg_num_new = toa_avg_num_new + 1;
				if (toa_avg_num_new > idx1.n_elem)
				{
					idx1_revise = find(COD_dem2_revise >= 0); //�൱�ڲ���������������idx1����
					break;
				}
				idx1 = idx1.head(toa_avg_num_new);

				COD_dem2_revise = COD_dem2.rows(idx1);
				idx1_revise = find(COD_dem2_revise <= 1);
			}

			idx1 = idx1.rows(idx1_revise);

			//idx1��blue band��idx2�ཻ
			uvec idx3 = intersect(idx1, idx2);
			if (idx3.n_elem > 4)
			{
				idx2 = idx3;
			}
			else
			{
				idx2 = idx1;
			}

			//idx_total = idx2; //�ܷ��䣬ln+cos,blue,����
			idx_total = idx1;
			idx_dir = idx1;//ֱ����䣬ln+cos
		}
		else if (clear_flag_dem2 == 3)//����
		{
			////��״
			////��������ֵ
			float toa_revise0 = mean(toa_rad_cloud);  //�۲�ֵ��ֵ	

			//����
			fvec cosine_multiband_dem2(toa_rad_lut_dem2_cloud.n_rows, arma::fill::zeros);
			fmat toa_rad_lut_dem2_cloud_log = log(toa_rad_lut_dem2_cloud);
			for (uword i = 0; i < toa_rad_lut_dem2_cloud.n_rows; i++)
			{

				cosine_multiband_dem2(i) = norm_dot(toa_rad_lut_dem2_cloud_log.row(i) - log(toa_revise0), trans(log(toa_rad_cloud)) - log(toa_revise0));

			}
			uvec idx1 = sort_index(cosine_multiband_dem2, "descend");
			idx1 = idx1.head(toa_avg_num);

			idx_total = idx1;
			idx_dir = idx1;
		}
		else //���ƺͲ�ȷ��
		{
			//�������ƶ�
			//��������ֵ
			float toa_revise0 = mean(toa_rad);  //�۲�ֵ��ֵ	

			fvec cosine_multiband_dem2(toa_rad_lut_dem2.n_rows, arma::fill::zeros);
			fmat toa_rad_lut_dem2_log = log(toa_rad_lut_dem2);

			for (uword i = 0; i < toa_rad_lut_dem2.n_rows; i++)
			{
				cosine_multiband_dem2(i) = norm_dot(toa_rad_lut_dem2_log.row(i) - log(toa_revise0), trans(log(toa_rad)) - log(toa_revise0));
			}

			uvec idx1 = sort_index(cosine_multiband_dem2, "descend");
			idx1 = idx1.head(toa_avg_num);
			//====================================
			if ((ref_band3 < 0.82)&(ref_band3 >= 0.65))
			{
				//��������ֵ
				toa_revise0 = mean(toa_rad_cloud);  //�۲�ֵ��ֵ	

				//����
				fvec cosine_multiband_dem2_2(toa_rad_lut_dem2_cloud.n_rows, arma::fill::zeros);
				fmat toa_rad_lut_dem2_log_2 = log(toa_rad_lut_dem2_cloud);

				for (uword i = 0; i < toa_rad_lut_dem2_cloud.n_rows; i++)
				{

					cosine_multiband_dem2_2(i) = norm_dot(toa_rad_lut_dem2_log_2.row(i) - log(toa_revise0), trans(log(toa_rad_cloud)) - log(toa_revise0));

				}
				idx1 = sort_index(cosine_multiband_dem2_2, "descend");
				idx1 = idx1.head(toa_avg_num);
			}
			//====================================

			//�ж�idx1��60����
			fvec COD_dem2_revise = COD_dem2.rows(idx1);

			uvec idx1_revise;
			idx1_revise = find(COD_dem2_revise <= 60 && COD_dem2_revise >= 0);

			int toa_avg_num_new = toa_avg_num;
			while (idx1_revise.n_elem < 2)
			{
				idx1 = sort_index(cosine_multiband_dem2, "descend");
				toa_avg_num_new = toa_avg_num_new + 1;
				if (toa_avg_num_new > idx1.n_elem)
				{
					idx1_revise = find(COD_dem2_revise >= 0); //�൱�ڲ���������������idx1����
					break;
				}
				idx1 = idx1.head(toa_avg_num_new);

				COD_dem2_revise = COD_dem2.rows(idx1);
				idx1_revise = find(COD_dem2_revise <= 60 && COD_dem2_revise >= 0);
			}

			idx1 = idx1.rows(idx1_revise);

			//�ཻ
			uvec idx3 = intersect(idx1, idx2);
			if (idx3.n_elem > 4)
			{
				idx2 = idx3;
			}

			COD_dem2_revise = COD_dem2.rows(idx2);
			uvec idx2_revise = find(COD_dem2_revise <= 60 && COD_dem2_revise >= 0);

			if (idx2_revise.n_elem != 0)
			{
				idx2 = idx2.rows(idx2_revise);
			}
			else
			{
				idx2 = idx1;
			}

			//idx_total = idx2; //ln+cos,blue,����
			idx_total = idx1;
			idx_dir = idx1; //ln+cos

		}

		//========================================================================================================
		if ((ref_mean < 0.3) && (clear_flag_dem2 != 3))
		{
			idx_total = idx2;
		}
		//----swdr-dem2----
		fvec swdr_dem2 = swdr_lut_dem2(idx_total);
		fvec rho_dem2 = rho_lut_dem2(idx_total);
		//
		vmean = mean(swdr_dem2);
		vstd = stddev(swdr_dem2) * f_std;
		idx = find(swdr_dem2 <= (vmean + vstd + 0.1) && swdr_dem2 >= (vmean - vstd - 0.1));
		if (idx.n_elem == 0)
		{
			cout << "(2) Interpolation - dw_DEM\n";
			cout << "[Error] cannot find valid elements for SWDR range from "
				<< vmean - vstd << " to " << vmean + vstd << endl;
			return 1;
		}
		float swdr_avg2 = mean(swdr_dem2(idx));
		float rho_avg2 = mean(rho_dem2(idx));
		finded_swdr_sub_v_dem2(i) = swdr_avg2;
		finded_rho_sub_v_dem2(i) = rho_avg2;

		//------------------------------------
		//---dir_dem2----
		swdr_dem2 = swdr_lut_dem2(idx_dir);
		fvec dir_dem2 = swdr_dir_lut_dem2(idx_dir);

		vmean = mean(swdr_dem2);
		vstd = stddev(swdr_dem2) * f_std;

		idx = find(swdr_dem2 <= (vmean + vstd + 0.1) && swdr_dem2 >= (vmean - vstd - 0.1));
		if (idx.n_elem == 0)
		{
			std::cout << "[Error_dem2] cannot find valid elements for SWDR(dir) range from "
				<< vmean - vstd << " to " << vmean + vstd << std::endl;
			return 1;
		}
		float dir_avg2 = mean(dir_dem2(idx));
		finded_swdr_dir_sub_v_dem2(i) = dir_avg2;

		//------------------------------------
		//---par_dem2----
		fvec par_dem2 = par_lut_dem2(idx_total);

		vmean = mean(par_dem2);
		vstd = stddev(par_dem2) * f_std;

		idx = find(par_dem2 <= (vmean + vstd + 0.1) && par_dem2 >= (vmean - vstd - 0.1));
		if (idx.n_elem == 0)
		{
			std::cout << "[Error_dem2] cannot find valid elements for PAR range from "
				<< vmean - vstd << " to " << vmean + vstd << std::endl;
			return 1;
		}
		float par_avg2 = mean(par_dem2(idx));
		finded_par_sub_v_dem2(i) = par_avg2;

		//------------------------------------
		//---par_dir_dem2----
		par_dem2 = par_lut_dem2(idx_dir);
		fvec par_dir_dem2 = par_dir_lut_dem2(idx_dir);

		vmean = mean(par_dem2);
		vstd = stddev(par_dem2) * f_std;

		idx = find(par_dem2 <= (vmean + vstd + 0.1) && par_dem2 >= (vmean - vstd - 0.1));
		if (idx.n_elem == 0)
		{
			std::cout << "[Error_dem2] cannot find valid elements for PAR(dir) range from "
				<< vmean - vstd << " to " << vmean + vstd << std::endl;
			return 1;
		}
		float par_dir_avg2 = mean(par_dir_dem2(idx));
		finded_par_dir_sub_v_dem2(i) = par_dir_avg2;

		//------------------------------------
		//-uva_dem2--
		fvec uva_dem2 = uva_lut_dem2(idx_total);

		vmean = mean(uva_dem2);
		vstd = stddev(uva_dem2) * f_std;
		idx = find(uva_dem2 <= (vmean + vstd + 0.01) && uva_dem2 >= (vmean - vstd - 0.01));
		if (idx.n_elem == 0)
		{
			cout << "[Error_dem2] cannot find valid elements for UVA range from "
				<< vmean - vstd << " to " << vmean + vstd << endl;
			return 1;
		}
		float uva_avg2 = mean(uva_dem2(idx));
		finded_uva_sub_v_dem2(i) = uva_avg2;


		//------------------------------------
		//-uvb_dem2--
		fvec uvb_dem2 = uvb_lut_dem2(idx_total);

		vmean = mean(uvb_dem2);
		vstd = stddev(uvb_dem2) * f_std;
		idx = find(uvb_dem2 <= (vmean + vstd + 0.01) && uvb_dem2 >= (vmean - vstd - 0.01));
		if (idx.n_elem == 0)
		{
			cout << "[Error_dem2] cannot find valid elements for UVB range from "
				<< vmean - vstd << " to " << vmean + vstd << endl;
			return 1;
		}
		float uvb_avg2 = mean(uvb_dem2(idx));
		finded_uvb_sub_v_dem2(i) = uvb_avg2;

		//------------------------------------
		//-toa_albedo_dem2--
		fvec toa_up_flux_dem2 = toa_up_flux_lut_dem2(idx_total);

		vmean = mean(toa_up_flux_dem2);
		vstd = stddev(toa_up_flux_dem2) * f_std;
		idx = find(toa_up_flux_dem2 <= (vmean + vstd + 0.1) && toa_up_flux_dem2 >= (vmean - vstd - 0.1));
		if (idx.n_elem == 0)
		{
			cout << "[Error_dem2] cannot find valid elements for TOA_up_flux range from "
				<< vmean - vstd << " to " << vmean + vstd << endl;
			return 1;
		}
		float toa_albedo_avg2 = mean(toa_up_flux_dem2(idx));
		finded_toa_up_flux_sub_v_dem2(i) = toa_albedo_avg2;

	}
////----------------------------------------------

	fvec slope = (finded_swdr_sub_v_dem1 - finded_swdr_sub_v_dem2) / (m_up_dem - m_dw_dem);
	itp_swdr = finded_swdr_sub_v_dem2 + slope % (dem_sub_v - m_dw_dem);

	slope = (finded_swdr_dir_sub_v_dem1 - finded_swdr_dir_sub_v_dem2) / (m_up_dem - m_dw_dem);
	itp_swdir = finded_swdr_dir_sub_v_dem2 + slope % (dem_sub_v - m_dw_dem);

	slope = (finded_par_sub_v_dem1 - finded_par_sub_v_dem2) / (m_up_dem - m_dw_dem);
	itp_par = finded_par_sub_v_dem2 + slope % (dem_sub_v - m_dw_dem);

	slope = (finded_par_dir_sub_v_dem1 - finded_par_dir_sub_v_dem2) / (m_up_dem - m_dw_dem);
	itp_pardir = finded_par_dir_sub_v_dem2 + slope % (dem_sub_v - m_dw_dem);

	slope = (finded_uva_sub_v_dem1 - finded_uva_sub_v_dem2) / (m_up_dem - m_dw_dem);
	itp_uva = finded_uva_sub_v_dem2 + slope % (dem_sub_v - m_dw_dem);

	slope = (finded_uvb_sub_v_dem1 - finded_uvb_sub_v_dem2) / (m_up_dem - m_dw_dem);
	itp_uvb = finded_uvb_sub_v_dem2 + slope % (dem_sub_v - m_dw_dem);

	slope = (finded_toa_up_flux_sub_v_dem1 - finded_toa_up_flux_sub_v_dem2) / (m_up_dem - m_dw_dem);
	itp_toa_up_flux = finded_toa_up_flux_sub_v_dem2 + slope % (dem_sub_v - m_dw_dem);

	slope = (finded_rho_sub_v_dem1 - finded_rho_sub_v_dem2) / (m_up_dem - m_dw_dem);
	itp_rho = finded_rho_sub_v_dem2 + slope % (dem_sub_v - m_dw_dem);

	return 0;
}
