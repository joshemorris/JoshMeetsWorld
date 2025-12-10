//-----------------------------------------------------------------------------
// Copyright 2012 Masanori Morise, Copyright 2025 Josh Morris
// Author: mmorise [at] meiji.ac.jp (Masanori Morise)
// Modified: Josh Morris
// Last update: 2025/09/25
//
//
// Spectral envelope estimation on the basis of the idea of CheapTrick.
//-----------------------------------------------------------------------------
#include "world/cheaptrick.h"

#include <math.h>
#include <vector>
#include <algorithm>

#include "world/common.h"
#include "world/constantnumbers.h"
#include "world/matlabfunctions.h"

//-----------------------------------------------------------------------------
// CheapTrickProcessor C++ class implementation
//-----------------------------------------------------------------------------

CheapTrickProcessor::CheapTrickProcessor(int fs, const CheapTrickOption *option)
    : m_fs(fs),
      m_q1(option->q1),
      m_fft_size(option->fft_size),
      m_f0_floor(GetF0FloorForCheapTrick(fs, option->fft_size)),
      m_spectrum_size(m_fft_size / 2 + 1){
  // Initialize random number generator state.
  randn_reseed(&m_randn_state);

  // Initialize FFT structures.
  InitializeForwardRealFFT(m_fft_size, &m_forward_real_fft);
  InitializeInverseRealFFT(m_fft_size, &m_inverse_real_fft);

  // Allocate reusable buffers for liftering.
  m_smoothing_lifter = new double[m_fft_size];
  m_compensation_lifter = new double[m_fft_size];

  // For streaming applications, we pre-allocate the temporary buffers used in
  // GetWindowedWaveform. The largest window size is determined by the
  // lowest possible F0, which is f0_floor_.
  int max_half_window_length = matlab_round(1.5 * m_fs / m_f0_floor);
  m_max_window_length = max_half_window_length * 2 + 1;

  // Allocate reusable buffers for windowing.
  m_base_index = new int[m_max_window_length];
  m_safe_index = new int[m_max_window_length];
  m_window = new double[m_max_window_length];
  
  // Allocate the spectral envelope buffer.
  m_spectral_envelope = new double[m_spectrum_size];
}

CheapTrickProcessor::~CheapTrickProcessor() {
  // Destroy FFT structures to free memory.
  DestroyForwardRealFFT(&m_forward_real_fft);
  DestroyInverseRealFFT(&m_inverse_real_fft);
  
  // Free the allocated buffers.
  delete[] m_smoothing_lifter;
  delete[] m_compensation_lifter;

  // Free the windowing buffers.
  delete[] m_base_index;
  delete[] m_safe_index;
  delete[] m_window;
  
  // Free the spectral envelope buffer.
  delete[] m_spectral_envelope;
}

void CheapTrickProcessor::SmoothingWithRecovery(double f0,
                                                double *spectral_envelope) {
  // Calculate lifter values.
  m_smoothing_lifter[0] = 1.0;
  m_compensation_lifter[0] = (1.0 - 2.0 * m_q1) + 2.0 * m_q1;
  double quefrency;
  for (int i = 1; i <= m_fft_size / 2; ++i) {
    quefrency = static_cast<double>(i) / m_fs;
    m_smoothing_lifter[i] = sin(world::kPi * f0 * quefrency) /
      (world::kPi * f0 * quefrency);
    m_compensation_lifter[i] = (1.0 - 2.0 * m_q1) + 2.0 * m_q1 *
      cos(2.0 * world::kPi * quefrency * f0);
  }

  // Go to cepstrum domain.
  for (int i = 0; i <= m_fft_size / 2; ++i)
    m_forward_real_fft.waveform[i] = log(m_forward_real_fft.waveform[i]);
  for (int i = 1; i < m_fft_size / 2; ++i)
    m_forward_real_fft.waveform[m_fft_size - i] = m_forward_real_fft.waveform[i];
  fft_execute(m_forward_real_fft.forward_fft);

  // Apply lifters in the cepstrum domain.
  for (int i = 0; i <= m_fft_size / 2; ++i) {
    m_inverse_real_fft.spectrum[i][0] = m_forward_real_fft.spectrum[i][0] *
      m_smoothing_lifter[i] * m_compensation_lifter[i] / m_fft_size;
    m_inverse_real_fft.spectrum[i][1] = 0.0;
  }

  // Go back to the spectral domain.
  fft_execute(m_inverse_real_fft.inverse_fft);

  for (int i = 0; i <= m_fft_size / 2; ++i)
    spectral_envelope[i] = exp(m_inverse_real_fft.waveform[i]);
}

void CheapTrickProcessor::GetPowerSpectrum(double f0) {
  int half_window_length = matlab_round(1.5 * m_fs / f0);

  // Zero-pad the waveform outside the window.
  for (int i = half_window_length * 2 + 1; i < m_fft_size; ++i)
    m_forward_real_fft.waveform[i] = 0.0;
  
  fft_execute(m_forward_real_fft.forward_fft);

  // Calculate the power spectrum.
  double *power_spectrum = m_forward_real_fft.waveform;
  for (int i = 0; i <= m_fft_size / 2; ++i)
    power_spectrum[i] =
      m_forward_real_fft.spectrum[i][0] * m_forward_real_fft.spectrum[i][0] +
      m_forward_real_fft.spectrum[i][1] * m_forward_real_fft.spectrum[i][1];

  // Apply DC correction.
  DCCorrection(power_spectrum, f0, m_fs, m_fft_size, power_spectrum);
}

void CheapTrickProcessor::GetWindowedWaveform(const double *x, int x_length,
                                              double current_f0,
                                              double current_position) {
  int half_window_length = matlab_round(1.5 * m_fs / current_f0);
  int window_length = half_window_length * 2 + 1;

  // These buffers are now pre-allocated in the constructor for streaming.
  // We use the member variables m_base_index, m_safe_index, and m_window instead.
  // Note: window_length is guaranteed to be <= m_max_window_length.

  // Set up indices and window function.
  for (int i = -half_window_length; i <= half_window_length; ++i)
    m_base_index[i + half_window_length] = i;
  int origin = matlab_round(current_position * m_fs + 0.001);
  for (int i = 0; i < window_length; ++i)
    m_safe_index[i] =
      MyMinInt(x_length - 1, MyMaxInt(0, origin + m_base_index[i]));

  double average = 0.0;
  double position;
  for (int i = 0; i < window_length; ++i) {
    position = m_base_index[i] / 1.5 / m_fs;
    m_window[i] = 0.5 * cos(world::kPi * position * current_f0) + 0.5;
    average += m_window[i] * m_window[i];
  }
  average = sqrt(average);
  for (int i = 0; i < window_length; ++i) m_window[i] /= average;

  // Apply the F0-adaptive window.
  double *waveform = m_forward_real_fft.waveform;
  for (int i = 0; i < window_length; ++i)
    waveform[i] = x[m_safe_index[i]] * m_window[i] +
      randn(&m_randn_state) * world::kMySafeGuardMinimum;

  double tmp_weight1 = 0;
  double tmp_weight2 = 0;
  for (int i = 0; i < window_length; ++i) {
    tmp_weight1 += waveform[i];
    tmp_weight2 += m_window[i];
  }
  double weighting_coefficient = tmp_weight1 / tmp_weight2;
  for (int i = 0; i < window_length; ++i)
    waveform[i] -= m_window[i] * weighting_coefficient;
}

void CheapTrickProcessor::AddInfinitesimalNoise() {
  for (int i = 0; i <= m_fft_size / 2; ++i)
    m_forward_real_fft.waveform[i] +=
        fabs(randn(&m_randn_state)) * world::kEps;
}

void CheapTrickProcessor::CheapTrickGeneralBody(const double *x, int x_length,
                                                double current_f0,
                                                double current_position) {
  // 1. F0-adaptive windowing. The result is stored in m_forward_real_fft.waveform.
  GetWindowedWaveform(x, x_length, current_f0, current_position);

  // 2. Calculate power spectrum with DC correction.
  // The result is stored back into m_forward_real_fft.waveform.
  GetPowerSpectrum(current_f0);

  // 3. Smooth the power spectrum on a linear axis.
  LinearSmoothing(m_forward_real_fft.waveform, current_f0 * 2.0 / 3.0,
      m_fs, m_fft_size, m_forward_real_fft.waveform);

  // 4. Add infinitesimal noise to avoid zeros in the spectrum.
  AddInfinitesimalNoise();

  // 5. Smooth on a log axis and apply spectral recovery in the cepstrum domain.
  // The result is stored in the member variable m_spectral_envelope.
  SmoothingWithRecovery(current_f0, m_spectral_envelope);
}

void CheapTrickProcessor::Process(const std::vector<double>& x,
                                  const std::vector<double>& temporal_positions,
                                  const std::vector<double>& f0,
                                  std::vector<std::vector<double>>& spectrogram) {
  int f0_length = static_cast<int>(f0.size());
  
  // Resize the output spectrogram vector to match dimensions.
  spectrogram.resize(f0_length, std::vector<double>(m_fft_size / 2 + 1));

  for (int i = 0; i < f0_length; ++i) {
    // Use default F0 for unvoiced frames below the floor.
    double current_f0 = f0[i] <= m_f0_floor ? world::kDefaultF0 : f0[i];
    
    // Process the current frame. The result is stored in m_spectral_envelope.
    // We pass .data() and .size() to the lower-level implementation.
    CheapTrickGeneralBody(x.data(), static_cast<int>(x.size()), current_f0,
        temporal_positions[i]);

    // Copy the result to the output spectrogram vector.
    for (int j = 0; j <= m_fft_size / 2; ++j)
      spectrogram[i][j] = m_spectral_envelope[j];
  }
}

//-----------------------------------------------------------------------------
// C-style API implementation (preserved for compatibility)
//-----------------------------------------------------------------------------

void CheapTrick(const double *x, int x_length, int fs,
    const double *temporal_positions, const double *f0, int f0_length,
    const CheapTrickOption *option, double **spectrogram) {
  
  // Create vectors from C-style arrays for compatibility with the new interface.
  std::vector<double> x_vec(x, x + x_length);
  std::vector<double> temporal_vec(temporal_positions, temporal_positions + f0_length);
  std::vector<double> f0_vec(f0, f0 + f0_length);
  std::vector<std::vector<double>> spectrogram_vec;

  // Instantiate the processor class.
  CheapTrickProcessor processor(fs, option);

  // Call the main processing method with vectors.
  processor.Process(x_vec, temporal_vec, f0_vec, spectrogram_vec);

  // Copy the result back to the C-style spectrogram buffer.
  for (int i = 0; i < f0_length; ++i) {
    std::copy(spectrogram_vec[i].begin(), spectrogram_vec[i].end(), spectrogram[i]);
  }
}

int GetFFTSizeForCheapTrick(int fs, const CheapTrickOption *option) {
  return static_cast<int>(pow(2.0, 1.0 +
      static_cast<int>(log(3.0 * fs / option->f0_floor + 1) / world::kLog2)));
}

double GetF0FloorForCheapTrick(int fs, int fft_size) {
  return 3.0 * fs / (fft_size - 3.0);
}

void InitializeCheapTrickOption(int fs, CheapTrickOption *option) {
  option->q1 = -0.15;
  option->f0_floor = world::kFloorF0;
  option->fft_size = GetFFTSizeForCheapTrick(fs, option);
}
