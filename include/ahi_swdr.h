#pragma once

#include "read_config_file.h"
#include <string>
#include <armadillo>

class ahi_swdr
{
public:
	ahi_swdr(const myConfig& cfg);
	~ahi_swdr();
	int sequential_run(const std::string& input_path, const std::string& output_path);

	int batch_run_tbb(const myConfig& cfg, const std::string& input_path, const std::string& output_path);
	int retrieve_image(const std::string& input_file, const std::string& out_file);
	//int retrieve_txt(const std::string& input_file, const std::string& out_file);

private:

	//void smooth_flux(arma::fmat& flux) const;

	int read_lut(const std::string& lut_file, const arma::uword& lut_cols);
	//int read_input_file(const std::string& input_file);

	//int save_result_file(const std::string& out_file, const arma::fmat& data);

	//int filter_sza(float sza, arma::uword& up_sza_idx, arma::uword& dw_sza_idx);
	//int filter_vza(float vza, arma::uword& up_vza_idx, arma::uword& dw_vza_idx);
	//int filter_los(float los, arma::uword& los_idx) const;
	//int filter_dem(float dem, arma::uword& up_dem_idx, arma::uword& dw_dem_idx);
	//int filter_lut(arma::mat& lut, arma::mat& lut_new,
	//	arma::uword& up_sza_idx, arma::uword& dw_sza_idx, arma::uword& up_vza_idx, arma::uword& dw_vza_idx,
	//	arma::uword& los_idx, arma::uword& up_dem_idx, arma::uword& dw_dem_idx);


	//int filter_lut(arma::fmat& lut, arma::fmat& lut_new);
	int filter_sza(float sza_min, float sza_max, arma::uword& up_sza_idx, arma::uword& dw_sza_idx, const arma::fvec& angle_list, arma::uword m_angle_min, arma::uword m_angle_max);
	// arma::fmat par_lut_tile, arma::fvec dir_par_lut_tile, arma::fmat uva_lut_tile,
	//arma::fvec dir_uva_lut_tile, arma::fmat uvb_lut_tile, arma::fvec dir_uvb_lut_tile, arma::fmat toa_up_flux_lut_tile,
	int interp_dem(arma::fvec& dem_sub_v, arma::fvec& toa_rad_b1_sub_v, arma::fvec& toa_rad_b3_sub_v, arma::fvec& toa_rad_b6_sub_v, arma::fvec& toa_rad_b7_sub_v,
		arma::fmat toa_rad_band1_lut_tile, arma::fmat toa_rad_band3_lut_tile, arma::fmat toa_rad_band6_lut_tile, arma::fmat toa_rad_band7_lut_tile,
		arma::fmat toa_rad_band1_lut_clear_tile, arma::fmat toa_rad_band3_lut_clear_tile, arma::fmat toa_rad_band6_lut_clear_tile, arma::fmat toa_rad_band7_lut_clear_tile,
		arma::fmat toa_rad_band1_lut_cloudy_tile, arma::fmat& toa_rad_band3_lut_cloudy_tile, arma::fmat toa_rad_band7_lut_cloudy_tile,
		arma::fmat toa_rad_ndsi_lut_tile, float lut_diff_max, float lut_diff_min,
		arma::uword& idx_up_dem, arma::uword& idx_dw_dem,
		arma::fmat swdr_lut_tile, arma::fvec swdr_dir_lut_tile, arma::fmat par_lut_tile, arma::fvec dir_par_lut_tile, arma::fmat uva_lut_tile,
		arma::fmat uvb_lut_tile, arma::fmat toa_up_flux_lut_tile,
		arma::fvec& COD, const arma::fvec& f_rho, arma::fvec& ref_mean_sub_v, arma::fvec& ref_band1_sub_v, arma::fvec& ref_band3_sub_v, arma::fvec& ref_band6_sub_v, arma::fvec& ref_band7_sub_v, arma::fvec& sza_sub_v,
		arma::fvec& itp_swdr, arma::fvec& itp_swdir, arma::fvec& itp_par, arma::fvec& itp_pardir,
		arma::fvec& itp_uva, arma::fvec& itp_uvb, arma::fvec& itp_toa_up_flux, arma::fvec& itp_rho) const;


	int get_SWDR(arma::fmat lut, arma::fvec& sza_sub_v, arma::fvec& vza_sub_v, arma::fvec& dem_sub_v,
		arma::fvec& toa_rad_b1_sub_v, arma::fvec& toa_rad_b3_sub_v, arma::fvec& toa_rad_b4_sub_v,
		arma::fvec& toa_rad_b6_sub_v, arma::fvec& toa_rad_b7_sub_v,
		arma::fvec& band1_ref_sub_v, arma::fvec& band3_ref_sub_v, arma::fvec& band4_ref_sub_v,
		arma::fvec& band6_ref_sub_v, arma::fvec& band7_ref_sub_v, arma::fvec& sw_albedo_sub_v, arma::fvec& vis_albedo_sub_v,
		float lut_diff_max, float lut_diff_min,
		arma::fvec& derived_swdr, arma::fvec& derived_dir, arma::fvec& derived_par, arma::fvec& derived_pardir,
		arma::fvec& derived_uva, arma::fvec& derived_uvb, 
		arma::fvec& derived_toa_up_flux, arma::fvec& derived_rho);

	int classify_atmos(
		arma::fmat& toa_rad_band1_lut, arma::fmat& toa_rad_band3_lut, arma::fmat& toa_rad_band6_lut, arma::fmat& toa_rad_band7_lut,
		arma::fmat& toa_rad_band1_lut_clear, arma::fmat& toa_rad_band3_lut_clear, arma::fmat& toa_rad_band6_lut_clear, arma::fmat& toa_rad_band7_lut_clear,
		arma::fmat& toa_rad_band1_lut_cloudy, arma::fmat& toa_rad_band3_lut_cloudy, arma::fmat& toa_rad_band7_lut_cloudy);

	const std::string m_lut_file;
	arma::fmat m_lut;
	arma::fmat m_input_records;

	const int m_toa_avg_num;
	const float m_ref_range;
	const int m_ref_bin_num;
	const float m_f_std;
	const int m_window;

	const arma::uvec m_sza_list;
	const arma::fvec m_sza_list_ft;
	arma::uword m_sza_min;
	arma::uword m_sza_max;

	const arma::uvec m_vza_list;
	const arma::fvec m_vza_list_ft;
	arma::uword m_vza_min;
	arma::uword m_vza_max;

	const arma::fvec m_dem_list;
	// const arma::fvec m_dem_list_ft;
	float m_min_dem;
	float m_max_dem;

	const arma::uvec m_los_list;
	const arma::fvec m_los_list_ft;

	arma::uword m_up_sza;
	arma::uword m_dw_sza;
	arma::uword m_up_vza;
	arma::uword m_dw_vza;
	float m_up_dem = 0;
	float m_dw_dem = 0;
	arma::uword m_up_dem10;
	arma::uword m_dw_dem10;
	arma::uword m_min_los;

	//控制LUT过滤参数
	arma::uword idx_filter_sza;
	arma::uword idx_filter_vza;
	arma::uword idx_filter_los;
	arma::uword idx_filter_dem;

	arma::uword lut_cols;
};