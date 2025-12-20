//-----------------------------------------------------------------------------
// Copyright 2012 Masanori Morise, Copyright 2025 Josh Morris
// Author: mmorise [at] meiji.ac.jp (Masanori Morise)
// Last update: 2025/12/20
//-----------------------------------------------------------------------------
#ifndef WORLD_SYNTHESISREALTIME_H_
#define WORLD_SYNTHESISREALTIME_H_

#include <vector>

#include "world/common.h"
#include "world/macrodefinitions.h"
#include "world/matlabfunctions.h"

WORLD_BEGIN_C_DECLS

//-----------------------------------------------------------------------------
// A struct for real-time synthesis.
// Maintained for backwards compatibility with the C API.
// The state is managed by the C++ class RealTimeSynthesizer (stored in 'impl').
//-----------------------------------------------------------------------------
typedef struct {
  // Basic parameters
  int fs;
  double frame_period;
  int buffer_size;
  int number_of_pointers;
  int fft_size;

  // Sound buffer for output. The length is buffer_size [sample].
  // This points to the memory managed by the C++ class.
  double *buffer;
  
  // State variables
  int current_pointer;
  int i;

  // For DC removal
  double *dc_remover;

  //---------------------------------------------------------------------------
  // Internal parameters exposed for legacy compatibility.
  // These point to the arrays managed by the C++ class.
  int *f0_length;
  int *f0_origin;
  double ***spectrogram;
  double ***aperiodicity;

  int current_pointer2;
  int head_pointer;
  int synthesized_sample;

  int handoff;
  double handoff_phase;
  double handoff_f0;
  int last_location;

  int cumulative_frame;
  int current_frame;

  double **interpolated_vuv;
  double **pulse_locations;
  int **pulse_locations_index;
  int *number_of_pulses;

  double *impulse_response;

  RandnState randn_state;

  // FFT structs
  MinimumPhaseAnalysis minimum_phase;
  InverseRealFFT inverse_real_fft;
  ForwardRealFFT forward_real_fft;

  // Pointer to the C++ RealTimeSynthesizer instance
  void *impl;
} WorldSynthesizer;

//-----------------------------------------------------------------------------
// InitializeSynthesizer() initializes the synthesizer.
// Internal: Creates a RealTimeSynthesizer C++ object and attaches it to synth->impl.
//-----------------------------------------------------------------------------
void InitializeSynthesizer(int fs, double frame_period, int fft_size,
  int buffer_size, int number_of_pointers, WorldSynthesizer *synth);

//-----------------------------------------------------------------------------
// AddParameters() attempts to add speech parameters.
//-----------------------------------------------------------------------------
int AddParameters(double *f0, int f0_length, double **spectrogram,
  double **aperiodicity, WorldSynthesizer *synth);

//-----------------------------------------------------------------------------
// RefreshSynthesizer() sets the parameters to default.
//-----------------------------------------------------------------------------
void RefreshSynthesizer(WorldSynthesizer *synth);

//-----------------------------------------------------------------------------
// DestroySynthesizer() releases the memory (deletes the C++ object).
//-----------------------------------------------------------------------------
void DestroySynthesizer(WorldSynthesizer *synth);

//-----------------------------------------------------------------------------
// IsLocked() checks whether the synthesizer is locked.
//-----------------------------------------------------------------------------
int IsLocked(WorldSynthesizer *synth);

//-----------------------------------------------------------------------------
// Synthesis2() generates speech with length of synth->buffer_size sample.
//-----------------------------------------------------------------------------
int Synthesis2(WorldSynthesizer *synth);

WORLD_END_C_DECLS

#ifdef __cplusplus
//-----------------------------------------------------------------------------
// C++ Class Refactoring
//-----------------------------------------------------------------------------
class RealTimeSynthesizer {
public:
  RealTimeSynthesizer(int fs, double frame_period, int fft_size, int buffer_size, int max_number_of_frames);
  ~RealTimeSynthesizer();

  int AddParameters(double *f0, int f0_length, double **spectrogram, double **aperiodicity);
  
  // High-level C++ interface
  // Takes input vectors, copies them to internal storage, and appends synthesized audio to output.
  // Returns true if successful, false if ring buffer is full or input is invalid.
  bool Process(const std::vector<double>& f0, 
               const std::vector<std::vector<double>>& spectrogram, 
               const std::vector<std::vector<double>>& aperiodicity, 
               std::vector<double>& output);

  // Renamed from Refresh to Reset as requested
  void Reset();
  
  void Destroy(); // Internal helper for cleanup
  
  // Renamed from IsLocked to better reflect the state being checked
  int IsRingBufferFull();
  
  int Synthesis();

  // Getters to link C struct fields to C++ members
  int GetFs() const { return mFs; }
  double GetFramePeriod() const { return mFramePeriod; }
  int GetBufferSize() const { return mBufferSize; }
  int GetMaxNumberOfFrames() const { return mMaxNumberOfFrames; }
  int GetFftSize() const { return mFftSize; }
  double* GetBuffer() const { return mBuffer; }
  double* GetDCRemover() const { return mDcRemover; }
  
  // Getters for internal array pointers (for legacy C struct compatibility)
  int* GetF0Length() const { return mF0Length; }
  int* GetF0Origin() const { return mF0Origin; }
  double*** GetSpectrogram() const { return mSpectrogram; }
  double*** GetAperiodicity() const { return mAperiodicity; }
  double** GetInterpolatedVuv() const { return mInterpolatedVuv; }
  double** GetPulseLocations() const { return mPulseLocations; }
  int** GetPulseLocationsIndex() const { return mPulseLocationsIndex; }
  int* GetNumberOfPulses() const { return mNumberOfPulses; }
  double* GetImpulseResponse() const { return mImpulseResponse; }
  
  // Accessors for syncing state integers
  int GetCurrentPointer() const { return mCurrentPointer; }
  int GetI() const { return mI; }
  int GetCurrentPointer2() const { return mCurrentPointer2; }
  int GetHeadPointer() const { return mHeadPointer; }
  int GetSynthesizedSample() const { return mSynthesizedSample; }
  int GetHandoff() const { return mHandoff; }
  double GetHandoffPhase() const { return mHandoffPhase; }
  double GetHandoffF0() const { return mHandoffF0; }
  int GetLastLocation() const { return mLastLocation; }
  int GetCumulativeFrame() const { return mCumulativeFrame; }
  int GetCurrentFrame() const { return mCurrentFrame; }

  // Struct syncing
  void SyncStruct(WorldSynthesizer* synth);

private:
  // Basic parameters
  int mFs;
  double mFramePeriod;
  int mBufferSize;
  int mMaxNumberOfFrames; // Renamed from mNumberOfPointers
  int mFftSize;

  // Constants for pre-allocation
  // Adjusted sizes to be consistent.
  // 1024 frames @ 5ms frame period = ~5.12 seconds
  // 262144 samples @ 48kHz = ~5.4 seconds
  static const int kMaxAllocatedSamples = 262144; 
  static const int kMaxAllocatedFrames = 1024;

  // Buffer
  double *mBuffer;

  // Pre-allocated scratch buffers for GetOneFrameSegment
  double *mAperiodicResponse;
  double *mPeriodicResponse;
  double *mSpectralEnvelope;
  double *mAperiodicRatio;

  // Pre-allocated scratch buffers for GetTimeBase and GetPulseLocationsForTimeBase
  double *mCoarseTimeAxis;
  double *mCoarseF0;
  double *mCoarseVuv;
  double *mInterpolatedF0;
  double *mTimeAxis;
  double *mTotalPhase;
  double *mWrapPhase;
  double *mWrapPhaseAbs;

  // Storage for Process() method inputs (to ensure data lifetime)
  // mSpecDataStore[pointer] -> flat array of data
  // mSpecPtrStore[pointer] -> array of pointers into data
  double **mSpecDataStore;
  double ***mSpecPtrStore;
  double **mAperDataStore;
  double ***mAperPtrStore;

  // Internal state
  int mCurrentPointer;
  int mI;
  double *mDcRemover;

  // Speech parameters in each pointer
  int *mF0Length;
  int *mF0Origin;
  double ***mSpectrogram;
  double ***mAperiodicity;

  int mCurrentPointer2;
  int mHeadPointer;
  int mSynthesizedSample;

  int mHandoff;
  double mHandoffPhase;
  double mHandoffF0;
  int mLastLocation;

  int mCumulativeFrame;
  int mCurrentFrame;

  // These are arrays of pointers, but the pointers now point to fixed pre-allocated buffers
  double **mInterpolatedVuv;
  double **mPulseLocations;
  int **mPulseLocationsIndex;
  int *mNumberOfPulses;

  double *mImpulseResponse;

  RandnState mRandnState;

  // FFT
  MinimumPhaseAnalysis mMinimumPhase;
  InverseRealFFT mInverseRealFFT;
  ForwardRealFFT mForwardRealFFT;

  // Internal helper methods (formerly static functions)
  void GetNoiseSpectrum(int noise_size, int fft_size, const ForwardRealFFT *forward_real_fft);
  void GetAperiodicResponse(int noise_size, int fft_size, const double *spectrum, 
      const double *aperiodic_ratio, double current_vuv, double *aperiodic_response);
  void ClearRingBuffer(int start, int end);
  int SeekSynthesizer(double current_location);
  void SearchPointer(int frame, int flag, double **front, double **next);
  void RemoveDCComponent(const double *periodic_response, int fft_size, 
      const double *dc_remover, double *new_periodic_response);
  void GetPeriodicResponse(int fft_size, const double *spectrum, 
      const double *aperiodic_ratio, double current_vuv, double *periodic_response);
  void GetSpectralEnvelope(double current_location, double *spectral_envelope);
  void GetAperiodicRatio(double current_location, double *aperiodic_spectrum);
  double GetCurrentVUV(int current_location);
  void GetOneFrameSegment(int noise_size, int current_location);
  void GetTemporalParametersForTimeBase(const double *f0, int f0_length, 
      double *coarse_time_axis, double *coarse_f0, double *coarse_vuv);
  void GetPulseLocationsForTimeBase(const double *interpolated_f0, 
      const double *time_axis, int number_of_samples, double origin);
  void GetTimeBase(const double *f0, int f0_length, int start_sample, int number_of_samples);
  int GetNextPulseLocationIndex();
  int UpdateSynthesizer(int current_location);
  int CheckSynthesizer();
  void GetDCRemover(int fft_size, double *dc_remover);
};
#endif // __cplusplus

#endif  // WORLD_SYNTHESISREALTIME_H_
