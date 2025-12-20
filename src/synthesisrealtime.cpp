//-----------------------------------------------------------------------------
// Copyright 2012 Masanori Morise, Copyright 2025 Josh Morris
// Author: mmorise [at] meiji.ac.jp (Masanori Morise)
// Last update: 2025/12/20
//
// Voice synthesis based on f0, spectrogram and aperiodicity.
// This is an implementation for real-time applications.
//
// Refactored to RealTimeSynthesizer C++ class.
//-----------------------------------------------------------------------------
#include "world/synthesisrealtime.h"

#include <math.h>
#include <stdlib.h>
#include <algorithm> // For std::copy

#include "world/common.h"
#include "world/constantnumbers.h"
#include "world/matlabfunctions.h"

//-----------------------------------------------------------------------------
// RealTimeSynthesizer Implementation
//-----------------------------------------------------------------------------

RealTimeSynthesizer::RealTimeSynthesizer(int fs, double frame_period, int fft_size, 
                                         int buffer_size, int max_number_of_frames)
    : mFs(fs), mFramePeriod(frame_period / 1000.0), mBufferSize(buffer_size),
      mMaxNumberOfFrames(max_number_of_frames), mFftSize(fft_size) {
  
  // Memory allocations
  mF0Length = new int[mMaxNumberOfFrames];
  mSpectrogram = new double**[mMaxNumberOfFrames];
  mAperiodicity = new double**[mMaxNumberOfFrames];
  mInterpolatedVuv = new double*[mMaxNumberOfFrames];
  mPulseLocations = new double*[mMaxNumberOfFrames];
  mPulseLocationsIndex = new int*[mMaxNumberOfFrames];
  mNumberOfPulses = new int[mMaxNumberOfFrames];
  mF0Origin = new int[mMaxNumberOfFrames];
  
  // Storage for Process() safety copies
  mSpecDataStore = new double*[mMaxNumberOfFrames];
  mSpecPtrStore = new double**[mMaxNumberOfFrames];
  mAperDataStore = new double*[mMaxNumberOfFrames];
  mAperPtrStore = new double**[mMaxNumberOfFrames];

  int spec_width = mFftSize / 2 + 1;

  for (int i = 0; i < mMaxNumberOfFrames; ++i) {
    mInterpolatedVuv[i] = new double[kMaxAllocatedSamples];
    mPulseLocations[i] = new double[kMaxAllocatedSamples];
    mPulseLocationsIndex[i] = new int[kMaxAllocatedSamples];

    // Allocate flat data buffers
    mSpecDataStore[i] = new double[kMaxAllocatedFrames * spec_width];
    mAperDataStore[i] = new double[kMaxAllocatedFrames * spec_width];
    
    // Allocate pointer arrays
    mSpecPtrStore[i] = new double*[kMaxAllocatedFrames];
    mAperPtrStore[i] = new double*[kMaxAllocatedFrames];

    // Pre-calculate pointers into the flat buffers
    for (int j = 0; j < kMaxAllocatedFrames; ++j) {
        mSpecPtrStore[i][j] = &mSpecDataStore[i][j * spec_width];
        mAperPtrStore[i][j] = &mAperDataStore[i][j * spec_width];
    }
  }

  mBuffer = new double[buffer_size * 2 + fft_size];
  mImpulseResponse = new double[mFftSize];
  mDcRemover = new double[mFftSize / 2];

  // Pre-allocate scratch buffers for GetOneFrameSegment
  mAperiodicResponse = new double[mFftSize];
  mPeriodicResponse = new double[mFftSize];
  mSpectralEnvelope = new double[mFftSize];
  mAperiodicRatio = new double[mFftSize];

  // Pre-allocate scratch buffers for GetTimeBase and GetPulseLocationsForTimeBase
  mCoarseTimeAxis = new double[kMaxAllocatedFrames];
  mCoarseF0 = new double[kMaxAllocatedFrames];
  mCoarseVuv = new double[kMaxAllocatedFrames];
  
  mInterpolatedF0 = new double[kMaxAllocatedSamples];
  mTimeAxis = new double[kMaxAllocatedSamples];
  
  mTotalPhase = new double[kMaxAllocatedSamples];
  mWrapPhase = new double[kMaxAllocatedSamples];
  mWrapPhaseAbs = new double[kMaxAllocatedSamples];

  // Initialize internal parameters
  Reset();

  InitializeMinimumPhaseAnalysis(mFftSize, &mMinimumPhase);
  InitializeInverseRealFFT(mFftSize, &mInverseRealFFT);
  InitializeForwardRealFFT(mFftSize, &mForwardRealFFT);
}

RealTimeSynthesizer::~RealTimeSynthesizer() {
  Destroy();
}

void RealTimeSynthesizer::Destroy() {
  delete[] mF0Length;
  delete[] mSpectrogram;
  delete[] mAperiodicity;

  delete[] mBuffer;
  delete[] mImpulseResponse;
  delete[] mDcRemover;

  // Delete ring buffer contents
  if (mInterpolatedVuv) {
    for (int i = 0; i < mMaxNumberOfFrames; ++i) {
      delete[] mInterpolatedVuv[i];
      delete[] mPulseLocations[i];
      delete[] mPulseLocationsIndex[i];
      
      delete[] mSpecDataStore[i];
      delete[] mSpecPtrStore[i];
      delete[] mAperDataStore[i];
      delete[] mAperPtrStore[i];
    }
  }

  delete[] mInterpolatedVuv;
  delete[] mPulseLocations;
  delete[] mPulseLocationsIndex;
  delete[] mNumberOfPulses;
  delete[] mF0Origin;

  delete[] mSpecDataStore;
  delete[] mSpecPtrStore;
  delete[] mAperDataStore;
  delete[] mAperPtrStore;

  // Delete scratch buffers
  delete[] mAperiodicResponse;
  delete[] mPeriodicResponse;
  delete[] mSpectralEnvelope;
  delete[] mAperiodicRatio;

  delete[] mCoarseTimeAxis;
  delete[] mCoarseF0;
  delete[] mCoarseVuv;
  delete[] mInterpolatedF0;
  delete[] mTimeAxis;
  delete[] mTotalPhase;
  delete[] mWrapPhase;
  delete[] mWrapPhaseAbs;

  DestroyMinimumPhaseAnalysis(&mMinimumPhase);
  DestroyInverseRealFFT(&mInverseRealFFT);
  DestroyForwardRealFFT(&mForwardRealFFT);
}

void RealTimeSynthesizer::Reset() {
  ClearRingBuffer(0, mMaxNumberOfFrames);
  mHandoffPhase = 0;
  mHandoffF0 = 0;
  mCumulativeFrame = -1;
  mLastLocation = 0;

  mCurrentPointer = 0;
  mCurrentPointer2 = 0;
  mHeadPointer = 0;
  mHandoff = 0;

  mI = 0;
  mCurrentFrame = 0;

  mSynthesizedSample = 0;

  for (int i = 0; i < mBufferSize * 2 + mFftSize; ++i)
    mBuffer[i] = 0;
  
  GetDCRemover(mFftSize / 2, mDcRemover);
  randn_reseed(&mRandnState);
}

bool RealTimeSynthesizer::Process(const std::vector<double>& f0, 
                                  const std::vector<std::vector<double>>& spectrogram, 
                                  const std::vector<std::vector<double>>& aperiodicity, 
                                  std::vector<double>& output) {
  if (f0.empty()) {
      // Nothing to add
      return false;
  }
  
  if (IsRingBufferFull()) {
      return false;
  }

  int frames = static_cast<int>(f0.size());
  if (frames > kMaxAllocatedFrames) {
      return false; // Input too large for internal buffer
  }

  // Determine current write pointer
  int pointer = mHeadPointer % mMaxNumberOfFrames;
  int spec_width = mFftSize / 2 + 1;

  // Copy data to internal managed storage
  // Note: We use the pre-calculated pointers in mSpecPtrStore/mAperPtrStore.
  // We just need to copy the data into the location those pointers point to.
  
  for (int i = 0; i < frames; ++i) {
      // Safety check for inner vector sizes
      if (spectrogram[i].size() < static_cast<size_t>(spec_width) || 
          aperiodicity[i].size() < static_cast<size_t>(spec_width)) {
          return false; 
      }
      
      // Copy to pre-allocated row
      // mSpecPtrStore[pointer][i] points to the i-th row in the flat buffer
      std::copy(spectrogram[i].begin(), spectrogram[i].begin() + spec_width, mSpecPtrStore[pointer][i]);
      std::copy(aperiodicity[i].begin(), aperiodicity[i].begin() + spec_width, mAperPtrStore[pointer][i]);
  }

  // Call AddParameters using the managed pointers
  // const_cast is needed because AddParameters signature is double* but we are passing data we own
  int result = AddParameters(const_cast<double*>(f0.data()), frames, mSpecPtrStore[pointer], mAperPtrStore[pointer]);
  
  if (result == 0) {
      return false;
  }

  // Loop Synthesis until we drain the synthesizer
  while (Synthesis()) {
      // Append generated samples to output vector
      output.insert(output.end(), mBuffer, mBuffer + mBufferSize);
  }

  return true;
}

void RealTimeSynthesizer::GetNoiseSpectrum(int noise_size, int fft_size,
    const ForwardRealFFT *forward_real_fft) {
  double average = 0.0;
  for (int i = 0; i < noise_size; ++i) {
    forward_real_fft->waveform[i] = randn(&mRandnState);
    average += forward_real_fft->waveform[i];
  }

  average /= noise_size;
  for (int i = 0; i < noise_size; ++i)
    forward_real_fft->waveform[i] -= average;
  for (int i = noise_size; i < fft_size; ++i)
    forward_real_fft->waveform[i] = 0.0;
  fft_execute(forward_real_fft->forward_fft);
}

void RealTimeSynthesizer::GetAperiodicResponse(int noise_size, int fft_size,
    const double *spectrum, const double *aperiodic_ratio, double current_vuv,
    double *aperiodic_response) {
  GetNoiseSpectrum(noise_size, fft_size, &mForwardRealFFT);

  if (current_vuv != 0.0)
    for (int i = 0; i <= mMinimumPhase.fft_size / 2; ++i)
      mMinimumPhase.log_spectrum[i] =
        log(spectrum[i] * aperiodic_ratio[i] +
        world::kMySafeGuardMinimum) / 2.0;
  else
    for (int i = 0; i <= mMinimumPhase.fft_size / 2; ++i)
      mMinimumPhase.log_spectrum[i] = log(spectrum[i]) / 2.0;
  GetMinimumPhaseSpectrum(&mMinimumPhase);

  for (int i = 0; i <= fft_size / 2; ++i) {
    mInverseRealFFT.spectrum[i][0] =
      mMinimumPhase.minimum_phase_spectrum[i][0] *
      mForwardRealFFT.spectrum[i][0] -
      mMinimumPhase.minimum_phase_spectrum[i][1] *
      mForwardRealFFT.spectrum[i][1];
    mInverseRealFFT.spectrum[i][1] =
      mMinimumPhase.minimum_phase_spectrum[i][0] *
      mForwardRealFFT.spectrum[i][1] +
      mMinimumPhase.minimum_phase_spectrum[i][1] *
      mForwardRealFFT.spectrum[i][0];
  }
  fft_execute(mInverseRealFFT.inverse_fft);
  fftshift(mInverseRealFFT.waveform, fft_size, aperiodic_response);
}

void RealTimeSynthesizer::ClearRingBuffer(int start, int end) {
  int pointer;
  for (int i = start; i < end; ++i) {
    pointer = i % mMaxNumberOfFrames;
    mNumberOfPulses[pointer] = 0;
    // Note: We do NOT delete the arrays here anymore as they are pre-allocated.
    // We just mark pulses as 0. 
  }
}

int RealTimeSynthesizer::SeekSynthesizer(double current_location) {
  int frame_number = static_cast<int>(current_location / mFramePeriod);

  int tmp_pointer = mCurrentPointer2;
  int tmp;
  for (int i = 0; i < mHeadPointer - mCurrentPointer2; ++i) {
    tmp = (tmp_pointer + i) % mMaxNumberOfFrames;
    if (mF0Origin[tmp] <= frame_number &&
        frame_number < mF0Origin[tmp] + mF0Length[tmp]) {
      tmp_pointer += i;
      break;
    }
  }
  ClearRingBuffer(mCurrentPointer2, tmp_pointer);
  mCurrentPointer2 = tmp_pointer;
  return 1;
}

void RealTimeSynthesizer::SearchPointer(int frame, int flag,
    double **front, double **next) {
  int pointer = mCurrentPointer2 % mMaxNumberOfFrames;
  int index = -1;
  for (int i = 0; i < mF0Length[pointer]; ++i)
    if (mF0Origin[pointer] + i == frame) {
      index = i;
      break;
    }

  double ***tmp_pointer =
    flag == 0 ? mSpectrogram : mAperiodicity;

  *front = tmp_pointer[pointer][index];
  *next = index == mF0Length[pointer] - 1 ?
    tmp_pointer[(mCurrentPointer2 + 1) %
    mMaxNumberOfFrames][0] : tmp_pointer[pointer][index + 1];
}

void RealTimeSynthesizer::RemoveDCComponent(const double *periodic_response, int fft_size,
    const double *dc_remover, double *new_periodic_response) {
  double dc_component = 0.0;
  for (int i = fft_size / 2; i < fft_size; ++i)
    dc_component += periodic_response[i];
  for (int i = 0; i < fft_size / 2; ++i)
    new_periodic_response[i] = 0.0;
  for (int i = fft_size / 2; i < fft_size; ++i)
    new_periodic_response[i] -= dc_component * dc_remover[i - fft_size / 2];
}

void RealTimeSynthesizer::GetPeriodicResponse(int fft_size, const double *spectrum,
    const double *aperiodic_ratio, double current_vuv, double *periodic_response) {
  if (current_vuv <= 0.5 || aperiodic_ratio[0] > 0.999) {
    for (int i = 0; i < fft_size; ++i) periodic_response[i] = 0.0;
    return;
  }

  for (int i = 0; i <= mMinimumPhase.fft_size / 2; ++i)
    mMinimumPhase.log_spectrum[i] =
      log(spectrum[i] * (1.0 - aperiodic_ratio[i]) +
      world::kMySafeGuardMinimum) / 2.0;
  GetMinimumPhaseSpectrum(&mMinimumPhase);

  for (int i = 0; i <= fft_size / 2; ++i) {
    mInverseRealFFT.spectrum[i][0] =
      mMinimumPhase.minimum_phase_spectrum[i][0];
    mInverseRealFFT.spectrum[i][1] =
      mMinimumPhase.minimum_phase_spectrum[i][1];
  }

  fft_execute(mInverseRealFFT.inverse_fft);
  fftshift(mInverseRealFFT.waveform, fft_size, periodic_response);
  RemoveDCComponent(periodic_response, fft_size, mDcRemover,
      periodic_response);
}

void RealTimeSynthesizer::GetSpectralEnvelope(double current_location, double *spectral_envelope) {
  int current_frame_floor =
    static_cast<int>(current_location / mFramePeriod);

  int current_frame_ceil =
    static_cast<int>(ceil(current_location / mFramePeriod));
  double interpolation =
    current_location / mFramePeriod - current_frame_floor;
  double *front = NULL;
  double *next = NULL;
  SearchPointer(current_frame_floor, 0, &front, &next);

  if (current_frame_floor == current_frame_ceil)
    for (int i = 0; i <= mFftSize / 2; ++i)
      spectral_envelope[i] = fabs(front[i]);
  else
    for (int i = 0; i <= mFftSize / 2; ++i)
      spectral_envelope[i] =
      (1.0 - interpolation) * fabs(front[i]) + interpolation * fabs(next[i]);
}

void RealTimeSynthesizer::GetAperiodicRatio(double current_location, double *aperiodic_spectrum) {
  int current_frame_floor =
    static_cast<int>(current_location / mFramePeriod);

  int current_frame_ceil =
    static_cast<int>(ceil(current_location / mFramePeriod));
  double interpolation =
    current_location / mFramePeriod - current_frame_floor;

  double *front = NULL;
  double *next = NULL;
  SearchPointer(current_frame_floor, 1, &front, &next);

  if (current_frame_floor == current_frame_ceil)
    for (int i = 0; i <= mFftSize / 2; ++i)
      aperiodic_spectrum[i] = pow(GetSafeAperiodicity(front[i]), 2.0);
  else
    for (int i = 0; i <= mFftSize / 2; ++i)
      aperiodic_spectrum[i] =
        pow((1.0 - interpolation) * GetSafeAperiodicity(front[i]) +
        interpolation * GetSafeAperiodicity(next[i]), 2.0);
}

double RealTimeSynthesizer::GetCurrentVUV(int current_location) {
  double current_vuv = 0.0;
  int pointer = mCurrentPointer % mMaxNumberOfFrames;

  int start_sample = MyMaxInt(0,
    static_cast<int>(ceil((mF0Origin[pointer] - 1) *
    mFramePeriod * mFs)));

  current_vuv =
    mInterpolatedVuv[pointer][current_location - start_sample + 1];
  return current_vuv;
}

void RealTimeSynthesizer::GetOneFrameSegment(int noise_size, int current_location) {
  // Uses pre-allocated member buffers:
  // mAperiodicResponse, mPeriodicResponse, mSpectralEnvelope, mAperiodicRatio

  double tmp_location = static_cast<double>(current_location) / mFs;
  SeekSynthesizer(tmp_location);
  GetSpectralEnvelope(tmp_location, mSpectralEnvelope);
  GetAperiodicRatio(tmp_location, mAperiodicRatio);

  double current_vuv = GetCurrentVUV(current_location);

  // Synthesis of the periodic response
  GetPeriodicResponse(mFftSize, mSpectralEnvelope, mAperiodicRatio,
      current_vuv, mPeriodicResponse);

  // Synthesis of the aperiodic response
  GetAperiodicResponse(noise_size, mFftSize, mSpectralEnvelope,
      mAperiodicRatio, current_vuv, mAperiodicResponse);

  double sqrt_noise_size = sqrt(static_cast<double>(noise_size));
  for (int i = 0; i < mFftSize; ++i)
    mImpulseResponse[i] =
    (mPeriodicResponse[i] * sqrt_noise_size + mAperiodicResponse[i]) /
      mFftSize;
}

void RealTimeSynthesizer::GetTemporalParametersForTimeBase(const double *f0, int f0_length,
    double *coarse_time_axis, double *coarse_f0, double *coarse_vuv) {
  int cumulative_frame = MyMaxInt(0, mCumulativeFrame - f0_length);
  coarse_f0[0] = mHandoffF0;
  coarse_time_axis[0] = cumulative_frame * mFramePeriod;
  coarse_vuv[0] = mHandoffF0 == 0 ? 0.0 : 1.0;
  for (int i = 0; i < f0_length; ++i) {
    coarse_time_axis[i + mHandoff] =
      (i + cumulative_frame + mHandoff) * mFramePeriod;
    coarse_f0[i + mHandoff] = f0[i];
    coarse_vuv[i + mHandoff] = f0[i] == 0.0 ? 0.0 : 1.0;
  }
}

void RealTimeSynthesizer::GetPulseLocationsForTimeBase(const double *interpolated_f0,
    const double *time_axis, int number_of_samples, double origin) {
  // Use pre-allocated buffers: mTotalPhase, mWrapPhase, mWrapPhaseAbs
  // We should ideally check number_of_samples + mHandoff <= kMaxAllocatedSamples

  mTotalPhase[0] = mHandoff == 1 ? mHandoffPhase :
    2.0 * world::kPi * interpolated_f0[0] / mFs;

  mTotalPhase[1] = mTotalPhase[0] + 2.0 * world::kPi *
    interpolated_f0[0] / mFs;
  for (int i = 1 + mHandoff; i < number_of_samples + mHandoff; ++i)
    mTotalPhase[i] = mTotalPhase[i - 1] +
      2.0 * world::kPi * interpolated_f0[i - mHandoff] / mFs;
  mHandoffPhase = mTotalPhase[number_of_samples - 1 + mHandoff];

  for (int i = 0; i < number_of_samples + mHandoff; ++i)
    mWrapPhase[i] = fmod(mTotalPhase[i], 2.0 * world::kPi);

  for (int i = 0; i < number_of_samples - 1 + mHandoff; ++i)
    mWrapPhaseAbs[i] = fabs(mWrapPhase[i + 1] - mWrapPhase[i]);

  int pointer = mHeadPointer % mMaxNumberOfFrames;
  int number_of_pulses = 0;
  for (int i = 0; i < number_of_samples - 1 + mHandoff; ++i)
    if (mWrapPhaseAbs[i] > world::kPi) {
      mPulseLocations[pointer][number_of_pulses] =
        time_axis[i] - static_cast<double>(mHandoff) / mFs;
      mPulseLocationsIndex[pointer][number_of_pulses] =
        matlab_round(mPulseLocations[pointer][number_of_pulses] *
            mFs);
      ++number_of_pulses;
    }
  mNumberOfPulses[pointer] = number_of_pulses;

  if (number_of_pulses != 0)
    mLastLocation =
      mPulseLocationsIndex[pointer][number_of_pulses - 1];
}

void RealTimeSynthesizer::GetTimeBase(const double *f0, int f0_length, int start_sample,
    int number_of_samples) {
  // Use pre-allocated buffers: mCoarseTimeAxis, mCoarseF0, mCoarseVuv
  // Use pre-allocated buffers: mInterpolatedF0, mTimeAxis
  
  // Note: ideally check f0_length + mHandoff <= kMaxAllocatedFrames
  // and number_of_samples <= kMaxAllocatedSamples

  GetTemporalParametersForTimeBase(f0, f0_length,
      mCoarseTimeAxis, mCoarseF0, mCoarseVuv);

  for (int i = 0; i < number_of_samples; ++i)
    mTimeAxis[i] = (i + start_sample) / static_cast<double>(mFs);

  int pointer = mHeadPointer % mMaxNumberOfFrames;
  
  // interp1 writes into mInterpolatedF0 and mInterpolatedVuv[pointer]
  interp1(mCoarseTimeAxis, mCoarseF0, f0_length + mHandoff, mTimeAxis,
    number_of_samples, mInterpolatedF0);
  interp1(mCoarseTimeAxis, mCoarseVuv, f0_length + mHandoff, mTimeAxis,
    number_of_samples, mInterpolatedVuv[pointer]);
    
  for (int i = 0; i < number_of_samples; ++i) {
    mInterpolatedVuv[pointer][i] =
      mInterpolatedVuv[pointer][i] > 0.5 ? 1.0 : 0.0;
    mInterpolatedF0[i] =
      mInterpolatedVuv[pointer][i] == 0.0 ?
      world::kDefaultF0 : mInterpolatedF0[i];
  }

  GetPulseLocationsForTimeBase(mInterpolatedF0, mTimeAxis, number_of_samples,
      mCoarseTimeAxis[0]);

  mHandoffF0 = mInterpolatedF0[number_of_samples - 1];
}

int RealTimeSynthesizer::GetNextPulseLocationIndex() {
  int pointer = mCurrentPointer % mMaxNumberOfFrames;
  if (mI < mNumberOfPulses[pointer] - 1)
    return mPulseLocationsIndex[pointer][mI + 1];
  else if (mCurrentPointer == mHeadPointer - 1)
    return 0;

  for (int i = 1; i < mMaxNumberOfFrames; ++i) {
    pointer = (i + mCurrentPointer) % mMaxNumberOfFrames;
    if (mNumberOfPulses[pointer] != 0)
      return mPulseLocationsIndex[pointer][0];
  }
  return 0;
}

int RealTimeSynthesizer::UpdateSynthesizer(int current_location) {
  int pointer = mCurrentPointer % mMaxNumberOfFrames;
  if (mI < mNumberOfPulses[pointer] - 1) {
    mI++;
    return 1;
  } else {
    if (mCurrentPointer == mHeadPointer - 1) return 0;
  }

  for (int i = 1; i < mMaxNumberOfFrames; ++i) {
    pointer = (i + mCurrentPointer) % mMaxNumberOfFrames;
    if (mNumberOfPulses[pointer] != 0) {
      mI = 0;
      mCurrentPointer += i;
      return 1;
    }
  }
  return 0;
}

int RealTimeSynthesizer::CheckSynthesizer() {
  if (mSynthesizedSample + mBufferSize >= mLastLocation)
    return 0;

  int pointer = mCurrentPointer % mMaxNumberOfFrames;
  while (mNumberOfPulses[pointer] == 0) {
    if (mCurrentPointer == mHeadPointer) break;
    mCurrentPointer++;
    pointer = mCurrentPointer % mMaxNumberOfFrames;
  }
  return 1;
}

void RealTimeSynthesizer::GetDCRemover(int fft_size, double *dc_remover) {
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

int RealTimeSynthesizer::AddParameters(double *f0, int f0_length, double **spectrogram,
    double **aperiodicity) {
  if (mHeadPointer - mCurrentPointer2 == mMaxNumberOfFrames)
    return 0;  // Since the queue is full, we cannot add the parameters.
  
  // Calculate potential sample usage
  // Recalculate start/end for buffer usage
   int start_sample =
    MyMaxInt(0, static_cast<int>(ceil((mCumulativeFrame + 1 - f0_length) *
      mFramePeriod * mFs)));
      
   // Need to check f0_length here because GetTimeBase will access mCoarseTimeAxis[f0_length + mHandoff]
   if ((f0_length + 2) > kMaxAllocatedFrames) {
        return 0;
   }

  int pointer = mHeadPointer % mMaxNumberOfFrames;
  mF0Length[pointer] = f0_length;
  mF0Origin[pointer] = mCumulativeFrame + 1;
  mCumulativeFrame += f0_length;

  mSpectrogram[pointer] = spectrogram;
  mAperiodicity[pointer] = aperiodicity;
  if (mCumulativeFrame < 1) {
    mHandoffF0 = f0[f0_length - 1];
    mNumberOfPulses[pointer] = 0;
    mHeadPointer++;
    mHandoff = 1;
    return 1;
  }

  // Recalculate start/end for buffer usage
   start_sample =
    MyMaxInt(0, static_cast<int>(ceil((mCumulativeFrame - f0_length) *
      mFramePeriod * mFs)));
  int end_sample =
    static_cast<int>(ceil((mCumulativeFrame) *
      mFramePeriod * mFs));
  int number_of_samples = end_sample - start_sample;

  // SAFETY CHECK for pre-allocated buffers
  if (number_of_samples > kMaxAllocatedSamples) {
      // Input too large for pre-allocated buffers.
      // Recover state
      mCumulativeFrame -= f0_length; 
      return 0; 
  }

  // No allocation here anymore. We use mInterpolatedVuv[pointer], etc.

  GetTimeBase(f0, f0_length, start_sample, number_of_samples);

  mHandoffF0 = f0[f0_length - 1];
  mHeadPointer++;
  mHandoff = 1;
  return 1;
}

int RealTimeSynthesizer::IsRingBufferFull() {
  int judge = 0;
  if (mHeadPointer - mCurrentPointer2 == mMaxNumberOfFrames)
    judge++;
  if (mSynthesizedSample + mBufferSize >= mLastLocation)
    judge++;

  return judge == 2 ? 1 : 0;
}

int RealTimeSynthesizer::Synthesis() {
  if (CheckSynthesizer() == 0)
    return 0;
  
  for (int i = 0; i < mBufferSize + mFftSize; ++i)
    mBuffer[i] = mBuffer[i + mBufferSize];

  int pointer = mCurrentPointer % mMaxNumberOfFrames;
  int noise_size, offset, tmp, index;
  int current_location = mPulseLocationsIndex[pointer][mI];
  
  while (current_location < mSynthesizedSample + mBufferSize) {
    tmp = GetNextPulseLocationIndex();
    noise_size = tmp - current_location;

    GetOneFrameSegment(noise_size, current_location);
    offset =
      current_location - mSynthesizedSample - mFftSize / 2 + 1;
    for (int i = MyMaxInt(0, -offset); i < mFftSize; ++i) {
      index = i + offset;
      mBuffer[index] += mImpulseResponse[i];
    }
    current_location = tmp;
    UpdateSynthesizer(current_location);
  }
  mSynthesizedSample += mBufferSize;
  SeekSynthesizer(mSynthesizedSample);
  return 1;
}

void RealTimeSynthesizer::SyncStruct(WorldSynthesizer* synth) {
  // Sync back state variables to the struct if necessary for legacy read-access
  synth->current_pointer = mCurrentPointer;
  synth->i = mI;
  synth->current_pointer2 = mCurrentPointer2;
  synth->head_pointer = mHeadPointer;
  synth->synthesized_sample = mSynthesizedSample;
  synth->handoff = mHandoff;
  synth->handoff_phase = mHandoffPhase;
  synth->handoff_f0 = mHandoffF0;
  synth->last_location = mLastLocation;
  synth->cumulative_frame = mCumulativeFrame;
  synth->current_frame = mCurrentFrame;
}

//-----------------------------------------------------------------------------
// C API Wrappers
//-----------------------------------------------------------------------------

void InitializeSynthesizer(int fs, double frame_period, int fft_size,
    int buffer_size, int number_of_pointers, WorldSynthesizer *synth) {
  RealTimeSynthesizer *obj = new RealTimeSynthesizer(fs, frame_period, fft_size, buffer_size, number_of_pointers);
  synth->impl = static_cast<void*>(obj);
  
  // Expose basic parameters via struct
  synth->fs = obj->GetFs();
  synth->frame_period = obj->GetFramePeriod(); // Note: obj returns seconds
  
  synth->buffer_size = obj->GetBufferSize();
  synth->number_of_pointers = obj->GetMaxNumberOfFrames();
  synth->fft_size = obj->GetFftSize();
  synth->buffer = obj->GetBuffer();
  synth->dc_remover = obj->GetDCRemover();

  // Link internal array pointers for legacy field access
  synth->f0_length = obj->GetF0Length();
  synth->f0_origin = obj->GetF0Origin();
  synth->spectrogram = obj->GetSpectrogram();
  synth->aperiodicity = obj->GetAperiodicity();
  synth->interpolated_vuv = obj->GetInterpolatedVuv();
  synth->pulse_locations = obj->GetPulseLocations();
  synth->pulse_locations_index = obj->GetPulseLocationsIndex();
  synth->number_of_pulses = obj->GetNumberOfPulses();
  synth->impulse_response = obj->GetImpulseResponse();

  // Initial sync of state
  obj->SyncStruct(synth);
}

int AddParameters(double *f0, int f0_length, double **spectrogram,
    double **aperiodicity, WorldSynthesizer *synth) {
  RealTimeSynthesizer *obj = static_cast<RealTimeSynthesizer*>(synth->impl);
  int result = obj->AddParameters(f0, f0_length, spectrogram, aperiodicity);
  obj->SyncStruct(synth);
  return result;
}

void RefreshSynthesizer(WorldSynthesizer *synth) {
  RealTimeSynthesizer *obj = static_cast<RealTimeSynthesizer*>(synth->impl);
  obj->Reset();
  obj->SyncStruct(synth);
}

void DestroySynthesizer(WorldSynthesizer *synth) {
  RealTimeSynthesizer *obj = static_cast<RealTimeSynthesizer*>(synth->impl);
  delete obj;
  synth->impl = NULL;
  synth->buffer = NULL;
}

int IsLocked(WorldSynthesizer *synth) {
  RealTimeSynthesizer *obj = static_cast<RealTimeSynthesizer*>(synth->impl);
  return obj->IsRingBufferFull();
}

int Synthesis2(WorldSynthesizer *synth) {
  RealTimeSynthesizer *obj = static_cast<RealTimeSynthesizer*>(synth->impl);
  int result = obj->Synthesis();
  obj->SyncStruct(synth);
  return result;
}
