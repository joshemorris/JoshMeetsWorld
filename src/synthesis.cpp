//-----------------------------------------------------------------------------
// Copyright 2012 Masanori Morise, Copyright 2025 Josh Morris
// Author: mmorise [at] meiji.ac.jp (Masanori Morise)
// Last update: 2025/10/09
//
// Voice synthesis based on f0, spectrogram and aperiodicity.
// forward_real_fft, inverse_real_fft and minimum_phase are used to speed up.
//-----------------------------------------------------------------------------
#include "world/synthesis.h"

#include <math.h>

#include "world/constantnumbers.h"
#include "world/matlabfunctions.h"

namespace {
// These helper functions do not depend on class state and can remain static.
// They are used by the private methods of the Synthesizer class.
void GetSpectralEnvelope(double current_time, double frame_period,
    int f0_length, const double * const *spectrogram, int fft_size,
    double *spectral_envelope) {
  int current_frame_floor = MyMinInt(f0_length - 1,
    static_cast<int>(floor(current_time / frame_period)));
  int current_frame_ceil = MyMinInt(f0_length - 1,
    static_cast<int>(ceil(current_time / frame_period)));
  double interpolation = current_time / frame_period - current_frame_floor;

  if (current_frame_floor == current_frame_ceil)
    for (int i = 0; i <= fft_size / 2; ++i)
      spectral_envelope[i] = fabs(spectrogram[current_frame_floor][i]);
  else
    for (int i = 0; i <= fft_size / 2; ++i)
      spectral_envelope[i] =
        (1.0 - interpolation) * fabs(spectrogram[current_frame_floor][i]) +
        interpolation * fabs(spectrogram[current_frame_ceil][i]);
}

void GetAperiodicRatio(double current_time, double frame_period,
    int f0_length, const double * const *aperiodicity, int fft_size,
    double *aperiodic_spectrum) {
  int current_frame_floor = MyMinInt(f0_length - 1,
    static_cast<int>(floor(current_time / frame_period)));
  int current_frame_ceil = MyMinInt(f0_length - 1,
    static_cast<int>(ceil(current_time / frame_period)));
  double interpolation = current_time / frame_period - current_frame_floor;

  if (current_frame_floor == current_frame_ceil)
    for (int i = 0; i <= fft_size / 2; ++i)
      aperiodic_spectrum[i] =
        pow(GetSafeAperiodicity(aperiodicity[current_frame_floor][i]), 2.0);
  else
    for (int i = 0; i <= fft_size / 2; ++i)
      aperiodic_spectrum[i] = pow((1.0 - interpolation) *
          GetSafeAperiodicity(aperiodicity[current_frame_floor][i]) +
          interpolation *
          GetSafeAperiodicity(aperiodicity[current_frame_ceil][i]), 2.0);
}

void GetDCRemover(int fft_size, double *dc_remover) {
  double dc_component = 0.0;
  for (int i = 0; i < fft_size / 2; ++i) {
    dc_remover[i] = 0.5 -
      0.5 * cos(2.0 * world::kPi * (i + 1.0) / (1.0 + fft_size));
    dc_remover[fft_size - i - 1] = dc_remover[i];
    dc_component += dc_remover[i] * 2.0;
  }
  for (int i = 0; i < fft_size / 2; ++i) {
    dc_remover[i] /= dc_component;
    dc_remover[fft_size - i - 1] = dc_remover[i];
  }
}

}  // namespace

Synthesizer::Synthesizer(int fs, double frame_period, int fft_size,
    int f0_length, int y_length) :
    m_fs(fs), m_frame_period(frame_period / 1000.0), m_fft_size(fft_size) {
  // Initialize FFT and random number generator structures
  InitializeMinimumPhaseAnalysis(m_fft_size, &m_minimum_phase);
  InitializeInverseRealFFT(m_fft_size, &m_inverse_real_fft);
  InitializeForwardRealFFT(m_fft_size, &m_forward_real_fft);
  randn_reseed(&m_randn_state);

  // Allocate all required memory upfront
  m_impulse_response = new double[m_fft_size];
  m_dc_remover = new double[m_fft_size];
  m_aperiodic_response = new double[m_fft_size];
  m_periodic_response = new double[m_fft_size];
  m_spectral_envelope = new double[m_fft_size + 2];
  m_aperiodic_ratio = new double[m_fft_size + 2];
  m_pulse_locations = new double[y_length];
  m_pulse_locations_index = new int[y_length];
  m_pulse_locations_time_shift = new double[y_length];
  m_interpolated_vuv = new double[y_length];
  m_time_axis = new double[y_length];
  m_interpolated_f0 = new double[y_length];
  m_total_phase = new double[y_length];
  m_wrap_phase = new double[y_length];
  m_wrap_phase_abs = new double[y_length - 1];
  m_coarse_time_axis = new double[f0_length + 1];
  m_coarse_f0 = new double[f0_length + 1];
  m_coarse_vuv = new double[f0_length + 1];

  // Pre-calculate the DC remover window
  GetDCRemover(m_fft_size, m_dc_remover);
}

Synthesizer::~Synthesizer() {
  // Deallocate all memory
  DestroyMinimumPhaseAnalysis(&m_minimum_phase);
  DestroyInverseRealFFT(&m_inverse_real_fft);
  DestroyForwardRealFFT(&m_forward_real_fft);

  delete[] m_impulse_response;
  delete[] m_dc_remover;
  delete[] m_aperiodic_response;
  delete[] m_periodic_response;
  delete[] m_spectral_envelope;
  delete[] m_aperiodic_ratio;
  delete[] m_pulse_locations;
  delete[] m_pulse_locations_index;
  delete[] m_pulse_locations_time_shift;
  delete[] m_interpolated_vuv;
  delete[] m_time_axis;
  delete[] m_interpolated_f0;
  delete[] m_total_phase;
  delete[] m_wrap_phase;
  delete[] m_wrap_phase_abs;
  delete[] m_coarse_time_axis;
  delete[] m_coarse_f0;
  delete[] m_coarse_vuv;
}

void Synthesizer::GetNoiseSpectrum(int noise_size) {
  double average = 0.0;
  for (int i = 0; i < noise_size; ++i) {
    m_forward_real_fft.waveform[i] = randn(&m_randn_state);
    average += m_forward_real_fft.waveform[i];
  }

  average /= noise_size;
  for (int i = 0; i < noise_size; ++i)
    m_forward_real_fft.waveform[i] -= average;
  for (int i = noise_size; i < m_fft_size; ++i)
    m_forward_real_fft.waveform[i] = 0.0;
  fft_execute(m_forward_real_fft.forward_fft);
}

void Synthesizer::GetAperiodicResponse(int noise_size,
    const double *spectrum, const double *aperiodic_ratio, double current_vuv) {
  GetNoiseSpectrum(noise_size);

  if (current_vuv != 0.0)
    for (int i = 0; i <= m_minimum_phase.fft_size / 2; ++i)
      m_minimum_phase.log_spectrum[i] =
        log(spectrum[i] * aperiodic_ratio[i]) / 2.0;
  else
    for (int i = 0; i <= m_minimum_phase.fft_size / 2; ++i)
      m_minimum_phase.log_spectrum[i] = log(spectrum[i]) / 2.0;
  GetMinimumPhaseSpectrum(&m_minimum_phase);

  for (int i = 0; i <= m_fft_size / 2; ++i) {
    m_inverse_real_fft.spectrum[i][0] =
      m_minimum_phase.minimum_phase_spectrum[i][0] *
      m_forward_real_fft.spectrum[i][0] -
      m_minimum_phase.minimum_phase_spectrum[i][1] *
      m_forward_real_fft.spectrum[i][1];
    m_inverse_real_fft.spectrum[i][1] =
      m_minimum_phase.minimum_phase_spectrum[i][0] *
      m_forward_real_fft.spectrum[i][1] +
      m_minimum_phase.minimum_phase_spectrum[i][1] *
      m_forward_real_fft.spectrum[i][0];
  }
  fft_execute(m_inverse_real_fft.inverse_fft);
  fftshift(m_inverse_real_fft.waveform, m_fft_size, m_aperiodic_response);
}

void Synthesizer::RemoveDCComponent(const double *periodic_response,
    double *new_periodic_response) {
  double dc_component = 0.0;
  for (int i = m_fft_size / 2; i < m_fft_size; ++i)
    dc_component += periodic_response[i];
  for (int i = 0; i < m_fft_size / 2; ++i)
    new_periodic_response[i] = -dc_component * m_dc_remover[i];
  for (int i = m_fft_size / 2; i < m_fft_size; ++i)
    new_periodic_response[i] -= dc_component * m_dc_remover[i];
}

void Synthesizer::GetSpectrumWithFractionalTimeShift(double coefficient) {
  double re, im, re2, im2;
  for (int i = 0; i <= m_fft_size / 2; ++i) {
    re = m_inverse_real_fft.spectrum[i][0];
    im = m_inverse_real_fft.spectrum[i][1];
    re2 = cos(coefficient * i);
    im2 = sqrt(1.0 - re2 * re2);

    m_inverse_real_fft.spectrum[i][0] = re * re2 + im * im2;
    m_inverse_real_fft.spectrum[i][1] = im * re2 - re * im2;
  }
}

void Synthesizer::GetPeriodicResponse(const double *spectrum,
    const double *aperiodic_ratio, double current_vuv,
    double fractional_time_shift) {
  if (current_vuv <= 0.5 || aperiodic_ratio[0] > 0.999) {
    for (int i = 0; i < m_fft_size; ++i) m_periodic_response[i] = 0.0;
    return;
  }

  for (int i = 0; i <= m_minimum_phase.fft_size / 2; ++i)
    m_minimum_phase.log_spectrum[i] =
      log(spectrum[i] * (1.0 - aperiodic_ratio[i]) +
      world::kMySafeGuardMinimum) / 2.0;
  GetMinimumPhaseSpectrum(&m_minimum_phase);

  for (int i = 0; i <= m_fft_size / 2; ++i) {
    m_inverse_real_fft.spectrum[i][0] =
      m_minimum_phase.minimum_phase_spectrum[i][0];
    m_inverse_real_fft.spectrum[i][1] =
      m_minimum_phase.minimum_phase_spectrum[i][1];
  }

  double coefficient =
    2.0 * world::kPi * fractional_time_shift * m_fs / m_fft_size;
  GetSpectrumWithFractionalTimeShift(coefficient);

  fft_execute(m_inverse_real_fft.inverse_fft);
  fftshift(m_inverse_real_fft.waveform, m_fft_size, m_periodic_response);
  RemoveDCComponent(m_periodic_response, m_periodic_response);
}

void Synthesizer::GetOneFrameSegment(double current_vuv, int noise_size,
    const double * const *spectrogram, const double * const *aperiodicity,
    int f0_length, double current_time, double fractional_time_shift,
    double *response) {
  GetSpectralEnvelope(current_time, m_frame_period, f0_length, spectrogram,
      m_fft_size, m_spectral_envelope);
  GetAperiodicRatio(current_time, m_frame_period, f0_length, aperiodicity,
      m_fft_size, m_aperiodic_ratio);

  GetPeriodicResponse(m_spectral_envelope, m_aperiodic_ratio, current_vuv,
      fractional_time_shift);

  GetAperiodicResponse(noise_size, m_spectral_envelope, m_aperiodic_ratio,
      current_vuv);

  double sqrt_noise_size = sqrt(static_cast<double>(noise_size));
  for (int i = 0; i < m_fft_size; ++i)
    response[i] =
      (m_periodic_response[i] * sqrt_noise_size + m_aperiodic_response[i]) /
      m_fft_size;
}

void Synthesizer::GetTemporalParametersForTimeBase(const double *f0,
    int f0_length, double lowest_f0, int y_length) {
  for (int i = 0; i < y_length; ++i)
    m_time_axis[i] = i / static_cast<double>(m_fs);
  for (int i = 0; i < f0_length; ++i) {
    m_coarse_time_axis[i] = i * m_frame_period;
    m_coarse_f0[i] = f0[i] < lowest_f0 ? 0.0 : f0[i];
    m_coarse_vuv[i] = m_coarse_f0[i] == 0.0 ? 0.0 : 1.0;
  }
  m_coarse_time_axis[f0_length] = f0_length * m_frame_period;
  m_coarse_f0[f0_length] = m_coarse_f0[f0_length - 1] * 2 -
    m_coarse_f0[f0_length - 2];
  m_coarse_vuv[f0_length] = m_coarse_vuv[f0_length - 1] * 2 -
    m_coarse_vuv[f0_length - 2];
}

int Synthesizer::GetPulseLocationsForTimeBase(
    const double *interpolated_f0, int y_length) {
  m_total_phase[0] = 2.0 * world::kPi * interpolated_f0[0] / m_fs;
  m_wrap_phase[0] = fmod(m_total_phase[0], 2.0 * world::kPi);
  for (int i = 1; i < y_length; ++i) {
    m_total_phase[i] = m_total_phase[i - 1] +
      2.0 * world::kPi * interpolated_f0[i] / m_fs;
    m_wrap_phase[i] = fmod(m_total_phase[i], 2.0 * world::kPi);
    m_wrap_phase_abs[i - 1] = fabs(m_wrap_phase[i] - m_wrap_phase[i - 1]);
  }

  int number_of_pulses = 0;
  for (int i = 0; i < y_length - 1; ++i) {
    if (m_wrap_phase_abs[i] > world::kPi) {
      m_pulse_locations[number_of_pulses] = m_time_axis[i];
      m_pulse_locations_index[number_of_pulses] = i;

      double y1 = m_wrap_phase[i] - 2.0 * world::kPi;
      double y2 = m_wrap_phase[i + 1];
      double x = -y1 / (y2 - y1);
      m_pulse_locations_time_shift[number_of_pulses] = x / m_fs;

      ++number_of_pulses;
    }
  }

  return number_of_pulses;
}

int Synthesizer::GetTimeBase(const double *f0, int f0_length,
    double lowest_f0, int y_length) {
  GetTemporalParametersForTimeBase(f0, f0_length, lowest_f0, y_length);

  interp1(m_coarse_time_axis, m_coarse_f0, f0_length + 1,
      m_time_axis, y_length, m_interpolated_f0);
  interp1(m_coarse_time_axis, m_coarse_vuv, f0_length + 1,
      m_time_axis, y_length, m_interpolated_vuv);

  for (int i = 0; i < y_length; ++i) {
    m_interpolated_vuv[i] = m_interpolated_vuv[i] > 0.5 ? 1.0 : 0.0;
    m_interpolated_f0[i] =
      m_interpolated_vuv[i] == 0.0 ? world::kDefaultF0 : m_interpolated_f0[i];
  }

  return GetPulseLocationsForTimeBase(m_interpolated_f0, y_length);
}

void Synthesizer::process(const double *f0, int f0_length,
    const double * const *spectrogram, const double * const *aperiodicity,
    int y_length, double *y) {
  for (int i = 0; i < y_length; ++i) y[i] = 0.0;

  int number_of_pulses = GetTimeBase(f0, f0_length,
      m_fs / m_fft_size + 1.0, y_length);

  int noise_size;
  int index, offset, lower_limit, upper_limit;
  for (int i = 0; i < number_of_pulses; ++i) {
    noise_size = m_pulse_locations_index[MyMinInt(number_of_pulses - 1, i + 1)] -
      m_pulse_locations_index[i];

    GetOneFrameSegment(m_interpolated_vuv[m_pulse_locations_index[i]],
        noise_size, spectrogram, aperiodicity, f0_length, m_pulse_locations[i],
        m_pulse_locations_time_shift[i], m_impulse_response);

    offset = m_pulse_locations_index[i] - m_fft_size / 2 + 1;
    lower_limit = MyMaxInt(0, -offset);
    upper_limit = MyMinInt(m_fft_size, y_length - offset);
    for (int j = lower_limit; j < upper_limit; ++j) {
      index = j + offset;
      y[index] += m_impulse_response[j];
    }
  }
}

// C-style API for backward compatibility
void Synthesis(const double *f0, int f0_length,
    const double * const *spectrogram, const double * const *aperiodicity,
    int fft_size, double frame_period, int fs, int y_length, double *y) {
  Synthesizer synthesizer(fs, frame_period, fft_size, f0_length, y_length);
  synthesizer.process(f0, f0_length, spectrogram, aperiodicity, y_length, y);
}

