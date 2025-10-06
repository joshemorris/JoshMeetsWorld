//-----------------------------------------------------------------------------
// Copyright 2025 Josh Morris
// Last update: 2025/10/06
//-----------------------------------------------------------------------------
#include "world/d4c.h"

#include <math.h>

#include <algorithm>
#include <vector>

#include "world/common.h"
#include "world/constantnumbers.h"
#include "world/matlabfunctions.h"

namespace world {

namespace {
// This helper remains a free static function as it is stateless.
void SetParametersForGetWindowedWaveform(int half_window_length, int x_length,
                                         double current_position, int fs,
                                         double current_f0, int window_type,
                                         double window_length_ratio,
                                         int *base_index, int *safe_index,
                                         double *window) {
  for (int i = -half_window_length; i <= half_window_length; ++i)
    base_index[i + half_window_length] = i;
  int origin = matlab_round(current_position * fs + 0.001);
  for (int i = 0; i <= half_window_length * 2; ++i)
    safe_index[i] =
        MyMinInt(x_length - 1, MyMaxInt(0, origin + base_index[i]));

  // Designing of the window function
  double position;
  if (window_type == world::kHanning) {  // Hanning window
    for (int i = 0; i <= half_window_length * 2; ++i) {
      position = (2.0 * base_index[i] / window_length_ratio) / fs;
      window[i] = 0.5 * cos(world::kPi * position * current_f0) + 0.5;
    }
  } else {  // Blackman window
    for (int i = 0; i <= half_window_length * 2; ++i) {
      position = (2.0 * base_index[i] / window_length_ratio) / fs;
      window[i] = 0.42 + 0.5 * cos(world::kPi * position * current_f0) +
                  0.08 * cos(world::kPi * position * current_f0 * 2);
    }
  }
}
}  // namespace

D4C::D4C(int fs, int f0_length, int fft_size)
    : m_fs(fs),
      m_f0_length(f0_length),
      m_fft_size(fft_size),
      m_fft_size_d4c(static_cast<int>(
          pow(2.0, 1.0 +
                       static_cast<int>(log(4.0 * fs / kFloorF0D4C + 1) /
                                        kLog2)))),
      m_number_of_aperiodicities(
          static_cast<int>(MyMinDouble(kUpperLimit, fs / 2.0 - kFrequencyInterval) /
                           kFrequencyInterval)),
      m_window_length(
          static_cast<int>(kFrequencyInterval * m_fft_size_d4c / fs) * 2 + 1),
      m_fft_size_love_train(static_cast<int>(
          pow(2.0, 1.0 +
                       static_cast<int>(log(3.0 * fs / 40.0 + 1) / kLog2)))) {
  // Initialize random number generator state
  randn_reseed(&m_randn_state);

  // Initialize FFT plans for the two different FFT sizes used
  InitializeForwardRealFFT(m_fft_size_d4c, &m_forward_real_fft);
  InitializeForwardRealFFT(m_fft_size_love_train, &m_forward_real_fft_love_train);

  // Allocate and pre-calculate buffers
  m_window.resize(m_window_length);
  NuttallWindow(m_window_length, m_window.data());

  m_aperiodicity0.resize(f0_length);

  m_coarse_aperiodicity.resize(m_number_of_aperiodicities + 2);
  m_coarse_frequency_axis.resize(m_number_of_aperiodicities + 2);
  for (int i = 0; i <= m_number_of_aperiodicities; ++i)
    m_coarse_frequency_axis[i] = i * kFrequencyInterval;
  m_coarse_frequency_axis[m_number_of_aperiodicities + 1] = fs / 2.0;

  m_frequency_axis.resize(fft_size / 2 + 1);
  for (int i = 0; i <= fft_size / 2; ++i)
    m_frequency_axis[i] = static_cast<double>(i) * m_fs / m_fft_size;

  // Resize temporary buffers used in helper methods
  const int half_fft_d4c = m_fft_size_d4c / 2 + 1;
  m_static_centroid.resize(half_fft_d4c);
  m_smoothed_power_spectrum.resize(half_fft_d4c);
  m_static_group_delay.resize(half_fft_d4c);
  m_centroid1.resize(half_fft_d4c);
  m_centroid2.resize(half_fft_d4c);
  m_tmp_real.resize(half_fft_d4c);
  m_tmp_imag.resize(half_fft_d4c);
  m_smoothed_group_delay.resize(half_fft_d4c);
  m_power_spectrum_coarse.resize(half_fft_d4c);
  m_power_spectrum_love_train.resize(m_fft_size_love_train);
}

D4C::~D4C() {
  DestroyForwardRealFFT(&m_forward_real_fft);
  DestroyForwardRealFFT(&m_forward_real_fft_love_train);
}

void D4C::process(const double *x, int x_length,
                  const double *temporal_positions, const double *f0,
                  const D4COption *option, double **aperiodicity) {
  InitializeAperiodicity(aperiodicity);

  // D4C Love Train (Aperiodicity at 0 Hz is given by a different algorithm)
  D4CLoveTrain(x, x_length, f0, temporal_positions);

  m_coarse_aperiodicity[0] = -60.0;
  m_coarse_aperiodicity[m_number_of_aperiodicities + 1] = -kMySafeGuardMinimum;

  for (int i = 0; i < m_f0_length; ++i) {
    if (f0[i] == 0 || m_aperiodicity0[i] <= option->threshold) continue;
    D4CGeneralBody(x, x_length, MyMaxDouble(kFloorF0D4C, f0[i]),
                   temporal_positions[i], &m_coarse_aperiodicity[1]);

    // Linear interpolation to convert coarse aperiodicity to spectral representation
    GetAperiodicity(m_coarse_aperiodicity.data(), aperiodicity[i]);
  }
}

void D4C::InitializeAperiodicity(double **aperiodicity) {
  for (int i = 0; i < m_f0_length; ++i)
    for (int j = 0; j < m_fft_size / 2 + 1; ++j)
      aperiodicity[i][j] = 1.0 - kMySafeGuardMinimum;
}

void D4C::GetAperiodicity(const double *coarse_aperiodicity,
                          double *aperiodicity) {
  interp1(m_coarse_frequency_axis.data(), coarse_aperiodicity,
          m_number_of_aperiodicities + 2, m_frequency_axis.data(),
          m_fft_size / 2 + 1, aperiodicity);
  for (int i = 0; i <= m_fft_size / 2; ++i)
    aperiodicity[i] = pow(10.0, aperiodicity[i] / 20.0);
}

void D4C::GetWindowedWaveform(const double *x, int x_length, double current_f0,
                              double current_position, int window_type,
                              double window_length_ratio, double *waveform) {
  int half_window_length =
      matlab_round(window_length_ratio * m_fs / current_f0 / 2.0);
  int window_size = half_window_length * 2 + 1;

  // These buffers are small and their size changes with f0, so stack
  // or heap allocation here is reasonable.
  std::vector<int> base_index(window_size);
  std::vector<int> safe_index(window_size);
  std::vector<double> window(window_size);

  SetParametersForGetWindowedWaveform(
      half_window_length, x_length, current_position, m_fs, current_f0,
      window_type, window_length_ratio, base_index.data(), safe_index.data(),
      window.data());

  // F0-adaptive windowing
  for (int i = 0; i < window_size; ++i)
    waveform[i] = x[safe_index[i]] * window[i] + randn(&m_randn_state) * kSafeGuardD4C;

  double tmp_weight1 = 0.0;
  double tmp_weight2 = 0.0;
  for (int i = 0; i < window_size; ++i) {
    tmp_weight1 += waveform[i];
    tmp_weight2 += window[i];
  }
  double weighting_coefficient = tmp_weight1 / tmp_weight2;
  for (int i = 0; i < window_size; ++i)
    waveform[i] -= window[i] * weighting_coefficient;
}

void D4C::GetCentroid(const double *x, int x_length, double current_f0,
                      double current_position, double *centroid) {
  for (int i = 0; i < m_fft_size_d4c; ++i) m_forward_real_fft.waveform[i] = 0.0;
  GetWindowedWaveform(x, x_length, current_f0, current_position, kBlackman,
                      4.0, m_forward_real_fft.waveform);

  double power = 0.0;
  int upper_bound = matlab_round(2.0 * m_fs / current_f0) * 2;
  for (int i = 0; i <= upper_bound; ++i)
    power += m_forward_real_fft.waveform[i] * m_forward_real_fft.waveform[i];
  for (int i = 0; i <= upper_bound; ++i)
    m_forward_real_fft.waveform[i] /= sqrt(power);

  fft_execute(m_forward_real_fft.forward_fft);
  for (int i = 0; i <= m_fft_size_d4c / 2; ++i) {
    m_tmp_real[i] = m_forward_real_fft.spectrum[i][0];
    m_tmp_imag[i] = m_forward_real_fft.spectrum[i][1];
  }

  for (int i = 0; i < m_fft_size_d4c; ++i)
    m_forward_real_fft.waveform[i] *= i + 1.0;
  fft_execute(m_forward_real_fft.forward_fft);
  for (int i = 0; i <= m_fft_size_d4c / 2; ++i)
    centroid[i] = m_forward_real_fft.spectrum[i][0] * m_tmp_real[i] +
                  m_tmp_imag[i] * m_forward_real_fft.spectrum[i][1];
}

void D4C::GetStaticCentroid(const double *x, int x_length, double current_f0,
                            double current_position, double *static_centroid) {
  GetCentroid(x, x_length, current_f0, current_position - 0.25 / current_f0,
              m_centroid1.data());
  GetCentroid(x, x_length, current_f0, current_position + 0.25 / current_f0,
              m_centroid2.data());

  for (int i = 0; i <= m_fft_size_d4c / 2; ++i)
    static_centroid[i] = m_centroid1[i] + m_centroid2[i];

  DCCorrection(static_centroid, current_f0, m_fs, m_fft_size_d4c, static_centroid);
}

void D4C::GetSmoothedPowerSpectrum(const double *x, int x_length,
                                   double current_f0, double current_position,
                                   double *smoothed_power_spectrum) {
  for (int i = 0; i < m_fft_size_d4c; ++i) m_forward_real_fft.waveform[i] = 0.0;
  GetWindowedWaveform(x, x_length, current_f0, current_position, kHanning, 4.0,
                      m_forward_real_fft.waveform);

  fft_execute(m_forward_real_fft.forward_fft);
  for (int i = 0; i <= m_fft_size_d4c / 2; ++i)
    smoothed_power_spectrum[i] =
        m_forward_real_fft.spectrum[i][0] * m_forward_real_fft.spectrum[i][0] +
        m_forward_real_fft.spectrum[i][1] * m_forward_real_fft.spectrum[i][1];
  DCCorrection(smoothed_power_spectrum, current_f0, m_fs, m_fft_size_d4c,
               smoothed_power_spectrum);
  LinearSmoothing(smoothed_power_spectrum, current_f0, m_fs, m_fft_size_d4c,
                  smoothed_power_spectrum);
}

void D4C::GetStaticGroupDelay(const double *static_centroid,
                              const double *smoothed_power_spectrum, double f0,
                              double *static_group_delay) {
  for (int i = 0; i <= m_fft_size_d4c / 2; ++i)
    static_group_delay[i] = static_centroid[i] / smoothed_power_spectrum[i];
  LinearSmoothing(static_group_delay, f0 / 2.0, m_fs, m_fft_size_d4c,
                  static_group_delay);

  LinearSmoothing(static_group_delay, f0, m_fs, m_fft_size_d4c,
                  m_smoothed_group_delay.data());

  for (int i = 0; i <= m_fft_size_d4c / 2; ++i)
    static_group_delay[i] -= m_smoothed_group_delay[i];
}

void D4C::GetCoarseAperiodicity(const double *static_group_delay,
                                double *coarse_aperiodicity) {
  int boundary = matlab_round(m_fft_size_d4c * 8.0 / m_window_length);
  int half_window_length = m_window_length / 2;

  for (int i = 0; i < m_fft_size_d4c; ++i) m_forward_real_fft.waveform[i] = 0.0;

  int center;
  for (int i = 0; i < m_number_of_aperiodicities; ++i) {
    center = static_cast<int>(kFrequencyInterval * (i + 1) * m_fft_size_d4c / m_fs);
    for (int j = 0; j <= half_window_length * 2; ++j)
      m_forward_real_fft.waveform[j] =
          static_group_delay[center - half_window_length + j] * m_window[j];
    fft_execute(m_forward_real_fft.forward_fft);
    for (int j = 0; j <= m_fft_size_d4c / 2; ++j)
      m_power_spectrum_coarse[j] =
          m_forward_real_fft.spectrum[j][0] * m_forward_real_fft.spectrum[j][0] +
          m_forward_real_fft.spectrum[j][1] * m_forward_real_fft.spectrum[j][1];
    std::sort(m_power_spectrum_coarse.begin(), m_power_spectrum_coarse.end());
    for (int j = 1; j <= m_fft_size_d4c / 2; ++j)
      m_power_spectrum_coarse[j] += m_power_spectrum_coarse[j - 1];
    coarse_aperiodicity[i] =
        10 * log10(m_power_spectrum_coarse[m_fft_size_d4c / 2 - boundary - 1] /
                   m_power_spectrum_coarse[m_fft_size_d4c / 2]);
  }
}

void D4C::D4CLoveTrain(const double *x, int x_length, const double *f0,
                       const double *temporal_positions) {
  double lowest_f0 = 40.0;
  for (int i = 0; i < m_f0_length; ++i) {
    if (f0[i] == 0.0) {
      m_aperiodicity0[i] = 0.0;
      continue;
    }
    m_aperiodicity0[i] =
        D4CLoveTrainSub(x, x_length, MyMaxDouble(f0[i], lowest_f0),
                        temporal_positions[i]);
  }
}

double D4C::D4CLoveTrainSub(const double *x, int x_length, double current_f0,
                            double current_position) {
  int boundary0 = static_cast<int>(ceil(100.0 * m_fft_size_love_train / m_fs));
  int boundary1 = static_cast<int>(ceil(4000.0 * m_fft_size_love_train / m_fs));
  int boundary2 = static_cast<int>(ceil(7900.0 * m_fft_size_love_train / m_fs));

  int window_length = matlab_round(1.5 * m_fs / current_f0) * 2 + 1;
  GetWindowedWaveform(x, x_length, current_f0, current_position, kBlackman, 3.0,
                      m_forward_real_fft_love_train.waveform);

  for (int i = window_length; i < m_fft_size_love_train; ++i)
    m_forward_real_fft_love_train.waveform[i] = 0.0;
  fft_execute(m_forward_real_fft_love_train.forward_fft);

  for (int i = 0; i <= boundary0; ++i) m_power_spectrum_love_train[i] = 0.0;
  for (int i = boundary0 + 1; i < m_fft_size_love_train / 2 + 1; ++i)
    m_power_spectrum_love_train[i] =
        m_forward_real_fft_love_train.spectrum[i][0] *
            m_forward_real_fft_love_train.spectrum[i][0] +
        m_forward_real_fft_love_train.spectrum[i][1] *
            m_forward_real_fft_love_train.spectrum[i][1];
  for (int i = boundary0; i <= boundary2; ++i)
    m_power_spectrum_love_train[i] += m_power_spectrum_love_train[i - 1];

  return m_power_spectrum_love_train[boundary1] /
         m_power_spectrum_love_train[boundary2];
}

void D4C::D4CGeneralBody(const double *x, int x_length, double current_f0,
                         double current_position, double *coarse_aperiodicity) {
  GetStaticCentroid(x, x_length, current_f0, current_position,
                    m_static_centroid.data());
  GetSmoothedPowerSpectrum(x, x_length, current_f0, current_position,
                           m_smoothed_power_spectrum.data());
  GetStaticGroupDelay(m_static_centroid.data(), m_smoothed_power_spectrum.data(),
                      current_f0, m_static_group_delay.data());

  GetCoarseAperiodicity(m_static_group_delay.data(), coarse_aperiodicity);

  // Revision of the result based on the F0
  for (int i = 0; i < m_number_of_aperiodicities; ++i)
    coarse_aperiodicity[i] =
        MyMinDouble(0.0, coarse_aperiodicity[i] + (current_f0 - 100) / 50.0);
}

}  // namespace world

//-----------------------------------------------------------------------------
// C API implementation
//-----------------------------------------------------------------------------

void D4C(const double *x, int x_length, int fs,
         const double *temporal_positions, const double *f0, int f0_length,
         int fft_size, const world::D4COption *option, double **aperiodicity) {
  world::D4C d4c_processor(fs, f0_length, fft_size);
  d4c_processor.process(x, x_length, temporal_positions, f0, option,
                        aperiodicity);
}

void InitializeD4COption(world::D4COption *option) {
  option->threshold = world::kThreshold;
}
