//-----------------------------------------------------------------------------
// Copyright 2012 Masanori Morise, Copyright 2025 Josh Morris
// Last update: 2025/10/06
//-----------------------------------------------------------------------------
#ifndef WORLD_D4C_H_
#define WORLD_D4C_H_

#include <algorithm>
#include <vector>

#include "world/common.h"
#include "world/macrodefinitions.h"
#include "world/matlabfunctions.h"

namespace world {

//-----------------------------------------------------------------------------
// D4C Option Struct (remains a C-style struct for API compatibility)
//-----------------------------------------------------------------------------
typedef struct {
  double threshold;
} D4COption;

//-----------------------------------------------------------------------------
// D4C Class
//
// Encapsulates the D4C algorithm for aperiodicity estimation.
// This class is designed to be instantiated once with parameters like
// sampling rate and FFT size. This allows for efficient, repeated processing
// of audio by pre-allocating all necessary memory in the constructor,
// avoiding reallocations during the main processing loop.
//-----------------------------------------------------------------------------
class D4C {
 public:
  // Constructor: Pre-allocates memory for processing.
  // fs: Sampling frequency (Hz).
  // fft_size: FFT size for the output aperiodicity.
  D4C(int fs, int fft_size);

  // Destructor: Frees allocated memory and FFT plans.
  ~D4C();

  // process() calculates the aperiodicity for the given audio signal.
  // x: Input signal.
  // temporal_positions: Time axis for each F0 frame (in seconds).
  // f0: F0 contour (in Hz).
  // option: D4C options (e.g., VUV threshold).
  // aperiodicity: 2D output buffer for aperiodicity [f0_length][fft_size/2+1].
  void process(const std::vector<double>& x,
               const std::vector<double>& temporal_positions,
               const std::vector<double>& f0,
               const D4COption *option, 
               std::vector<std::vector<double>>& aperiodicity);

 private:
  void GetWindowedWaveform(const double *x, int x_length, double current_f0,
                           double current_position, int window_type,
                           double window_length_ratio, double *waveform);
  void GetCentroid(const double *x, int x_length, double current_f0,
                   double current_position, double *centroid);
  void GetStaticCentroid(const double *x, int x_length, double current_f0,
                         double current_position, double *static_centroid);
  void GetSmoothedPowerSpectrum(const double *x, int x_length,
                                double current_f0, double current_position,
                                double *smoothed_power_spectrum);
  void GetStaticGroupDelay(const double *static_centroid,
                           const double *smoothed_power_spectrum, double f0,
                           double *static_group_delay);
  void GetCoarseAperiodicity(const double *static_group_delay,
                             double *coarse_aperiodicity);
  void D4CLoveTrain(const double *x, int x_length, const double *f0, int f0_length,
                    const double *temporal_positions);
  double D4CLoveTrainSub(const double *x, int x_length, double current_f0,
                         double current_position);
  void D4CGeneralBody(const double *x, int x_length, double current_f0,
                      double current_position, double *coarse_aperiodicity);
  void GetAperiodicity(const double *coarse_aperiodicity, double *aperiodicity);
  void InitializeAperiodicity(std::vector<std::vector<double>>& aperiodicity, int f0_length);

  // Member variables for configuration and state.
  const int m_fs;
  const int m_fft_size;

  // D4C specific parameters, calculated in the constructor.
  const int m_fft_size_d4c;
  const int m_number_of_aperiodicities;
  const int m_window_length;
  const int m_fft_size_love_train;

  ForwardRealFFT m_forward_real_fft;
  ForwardRealFFT m_forward_real_fft_love_train;
  RandnState m_randn_state;

  // Pre-allocated buffers to avoid runtime allocations.
  std::vector<double> m_window;
  std::vector<double> m_aperiodicity0;
  std::vector<double> m_coarse_aperiodicity;
  std::vector<double> m_coarse_frequency_axis;
  std::vector<double> m_frequency_axis;

  // Reusable temporary buffers for internal calculations.
  std::vector<double> m_static_centroid;
  std::vector<double> m_smoothed_power_spectrum;
  std::vector<double> m_static_group_delay;
  std::vector<double> m_centroid1;
  std::vector<double> m_centroid2;
  std::vector<double> m_tmp_real;
  std::vector<double> m_tmp_imag;
  std::vector<double> m_smoothed_group_delay;
  std::vector<double> m_power_spectrum_coarse;
  std::vector<double> m_power_spectrum_love_train;
};

}  // namespace world

//-----------------------------------------------------------------------------
// Preserved C API for backward compatibility.
// These functions wrap the C++ class.
//-----------------------------------------------------------------------------
WORLD_BEGIN_C_DECLS

void D4C(const double *x, int x_length, int fs,
         const double *temporal_positions, const double *f0, int f0_length,
         int fft_size, const world::D4COption *option, double **aperiodicity);

void InitializeD4COption(world::D4COption *option);

WORLD_END_C_DECLS

#endif  // WORLD_D4C_H_
