//-----------------------------------------------------------------------------
// Copyright 2012 Masanori Morise, Copyright 2025 Josh Morris
// Author: mmorise [at] meiji.ac.jp (Masanori Morise)
// Modified: Josh Morris
// Last update: 2025/09/25
//
//
//-----------------------------------------------------------------------------
#ifndef WORLD_CHEAPTRICK_H_
#define WORLD_CHEAPTRICK_H_

#include "world/common.h"
#include "world/macrodefinitions.h"
#include "world/matlabfunctions.h"

// The C-style interface is preserved for backward compatibility.
WORLD_BEGIN_C_DECLS

//-----------------------------------------------------------------------------
// Struct for CheapTrick
//-----------------------------------------------------------------------------
typedef struct {
  double q1;
  double f0_floor;
  int fft_size;
} CheapTrickOption;

//-----------------------------------------------------------------------------
// CheapTrick() calculates the spectrogram that consists of spectral envelopes
// estimated by CheapTrick.
//-----------------------------------------------------------------------------
void CheapTrick(const double *x, int x_length, int fs,
    const double *temporal_positions, const double *f0, int f0_length,
    const CheapTrickOption *option, double **spectrogram);

//-----------------------------------------------------------------------------
// InitializeCheapTrickOption sets the default parameters for CheapTrick.
//-----------------------------------------------------------------------------
void InitializeCheapTrickOption(int fs, CheapTrickOption *option);

//-----------------------------------------------------------------------------
// GetFFTSizeForCheapTrick() calculates the FFT size based on the sampling
// frequency and the lower limit of f0.
//-----------------------------------------------------------------------------
int GetFFTSizeForCheapTrick(int fs, const CheapTrickOption *option);

//-----------------------------------------------------------------------------
// GetF0FloorForCheapTrick() calculates the actual lower f0 limit for
// CheapTrick.
//-----------------------------------------------------------------------------
double GetF0FloorForCheapTrick(int fs, int fft_size);

WORLD_END_C_DECLS

//-----------------------------------------------------------------------------
// C++ Class Implementation for CheapTrick
// This class encapsulates the state and logic of the CheapTrick algorithm,
// handling all memory allocation for its internal buffers in the constructor.
//-----------------------------------------------------------------------------
class CheapTrickProcessor {
 public:
  // Constructor: Initializes all components and allocates memory.
  CheapTrickProcessor(int fs, const CheapTrickOption *option);

  // Destructor: Cleans up all allocated resources.
  ~CheapTrickProcessor();

  // Processes the entire signal to produce a spectrogram.
  void Process(const double *x, int x_length, const double *temporal_positions,
               const double *f0, int f0_length, double **spectrogram);

  // The main processing method for a single frame, corresponding to the
  // original CheapTrickGeneralBody function.
  void CheapTrickGeneralBody(const double *x, int x_length,
                             double current_f0, double current_position);

  // Getters for important parameters.
  double get_f0_floor() const { return m_f0_floor; }
  int get_fft_size() const { return m_fft_size; }

 private:
  // Private helper methods, refactored from static functions.
  void SmoothingWithRecovery(double f0, double *spectral_envelope);
  void GetPowerSpectrum(double f0);
  void GetWindowedWaveform(const double *x, int x_length, double current_f0,
                           double current_position);
  void AddInfinitesimalNoise();

  // Member variables to hold state and pre-allocated memory.
  const int m_fs;
  const double m_q1;
  const int m_fft_size;
  const double m_f0_floor;

  ForwardRealFFT m_forward_real_fft;
  InverseRealFFT m_inverse_real_fft;
  RandnState m_randn_state;
  
  // Reusable buffer for the spectral envelope of a single frame.
  double *m_spectral_envelope;

  // Reusable buffers for liftering, allocated in the constructor.
  double *m_smoothing_lifter;
  double *m_compensation_lifter;

  // Reusable buffers for windowing, allocated in the constructor for streaming.
  int m_max_window_length;
  int *m_base_index;
  int *m_safe_index;
  double *m_window;

  // Disallow copy and assignment to prevent resource management issues.
  CheapTrickProcessor(const CheapTrickProcessor&) = delete;
  CheapTrickProcessor& operator=(const CheapTrickProcessor&) = delete;
};

#endif  // WORLD_CHEAPTRICK_H_
