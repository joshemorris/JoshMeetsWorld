//-----------------------------------------------------------------------------
// Copyright 2012 Masanori Morise, Copyright 2025 Josh Morris
// Author: mmorise [at] meiji.ac.jp (Masanori Morise)
// Last update: 2025/10/09
//-----------------------------------------------------------------------------
#ifndef WORLD_SYNTHESIS_H_
#define WORLD_SYNTHESIS_H_

#include "world/common.h"
#include "world/macrodefinitions.h"
#include "world/matlabfunctions.h"

// The Synthesizer class encapsulates all the necessary components and buffers
// for speech synthesis. It is designed for real-time applications by
// pre-allocating all necessary memory in its constructor.
class Synthesizer {
 public:
  //---------------------------------------------------------------------------
  // The constructor initializes all components and allocates memory.
  // This should be called once before starting the audio processing.
  //
  // Input:
  //   fs           : Sampling frequency
  //   frame_period : Frame period in milliseconds
  //   fft_size     : FFT size
  //   f0_length    : The length of the F0 contour for buffer allocation
  //   y_length     : The length of the output waveform for buffer allocation
  //---------------------------------------------------------------------------
  Synthesizer(int fs, double frame_period, int fft_size, int f0_length,
      int y_length);

  //---------------------------------------------------------------------------
  // The destructor safely deallocates all memory.
  //---------------------------------------------------------------------------
  ~Synthesizer();

  //---------------------------------------------------------------------------
  // The process method synthesizes the voice in real-time.
  // This function is designed to be allocation-free.
  //
  // Input:
  //   f0           : F0 contour
  //   f0_length    : Length of F0
  //   spectrogram  : Spectrogram
  //   aperiodicity : Aperiodicity spectrogram
  //   y_length     : Length of the output signal
  // Output:
  //   y            : Synthesized speech signal
  //---------------------------------------------------------------------------
  void process(const double *f0, int f0_length,
      const double * const *spectrogram, const double * const *aperiodicity,
      int y_length, double *y);

 private:
  // Private methods that were formerly static functions in the original file.
  // These are the core building blocks of the synthesis process.
  void GetNoiseSpectrum(int noise_size);
  void GetAperiodicResponse(int noise_size, const double *spectrum,
      const double *aperiodic_ratio, double current_vuv);
  void GetPeriodicResponse(const double *spectrum,
      const double *aperiodic_ratio, double current_vuv,
      double fractional_time_shift);
  void GetOneFrameSegment(double current_vuv, int noise_size,
      const double * const *spectrogram, const double * const *aperiodicity,
      int f0_length, double current_time, double fractional_time_shift,
      double *response);
  int GetTimeBase(const double *f0, int f0_length, double lowest_f0,
      int y_length);
  void GetTemporalParametersForTimeBase(const double *f0, int f0_length,
      double lowest_f0, int y_length);
  int GetPulseLocationsForTimeBase(const double *interpolated_f0, int y_length);
  void GetSpectrumWithFractionalTimeShift(double coefficient);
  void RemoveDCComponent(const double *periodic_response,
      double *new_periodic_response);

  // Class member variables are prefixed with m_
  // Basic synthesis parameters
  const int m_fs;
  const double m_frame_period;
  const int m_fft_size;

  // FFT-related structures and random number generator state
  RandnState m_randn_state;
  MinimumPhaseAnalysis m_minimum_phase;
  InverseRealFFT m_inverse_real_fft;
  ForwardRealFFT m_forward_real_fft;

  // Pre-allocated buffers to avoid real-time memory allocation.
  // Their sizes are determined in the constructor.
  double *m_impulse_response;
  double *m_pulse_locations;
  int *m_pulse_locations_index;
  double *m_pulse_locations_time_shift;
  double *m_interpolated_vuv;
  double *m_dc_remover;

  // Buffers for GetOneFrameSegment
  double *m_aperiodic_response;
  double *m_periodic_response;
  double *m_spectral_envelope;
  double *m_aperiodic_ratio;

  // Buffers for GetTimeBase and its helpers
  double *m_time_axis;
  double *m_coarse_time_axis;
  double *m_coarse_f0;
  double *m_coarse_vuv;
  double *m_interpolated_f0;
  double *m_total_phase;
  double *m_wrap_phase;
  double *m_wrap_phase_abs;
};

//-----------------------------------------------------------------------------
// C API for backward compatibility.
//-----------------------------------------------------------------------------
WORLD_BEGIN_C_DECLS

void Synthesis(const double *f0, int f0_length,
    const double * const *spectrogram, const double * const *aperiodicity,
    int fft_size, double frame_period, int fs, int y_length, double *y);

WORLD_END_C_DECLS

#endif  // WORLD_SYNTHESIS_H_
