#include <iostream>
#include <string>
#include "feat/feature-fbank.h"
#include "matrix/kaldi-matrix.h"
#include "util/kaldi-io.h"
#include "base/kaldi-types.h"
#include "feat/wave-reader.h"
#include "transform/cmvn.h"

extern "C" {

// C-style interface for Fbank computation
int compute_fbank(const char* wav_path, int num_bins, float** output, int* rows, int* cols) {
    try {
        // Read waveform
        kaldi::Input ki(wav_path);
        kaldi::WaveData wave_data;
        wave_data.Read(ki.Stream());

        int32_t num_channels = wave_data.Data().NumRows();
        if (num_channels != 1) {
            std::cerr << "Input audio has more than one channel. Only mono audio is supported." << std::endl;
            return -1;
        }

        const kaldi::Matrix<kaldi::BaseFloat>& waveform = wave_data.Data();
        kaldi::SubVector<kaldi::BaseFloat> mono_waveform(waveform, 0);

        int sample_rate = wave_data.SampFreq();

        // Configure Fbank options
        kaldi::FbankOptions fbank_opts;
        fbank_opts.frame_opts.samp_freq = sample_rate;
        fbank_opts.mel_opts.num_bins = num_bins;

        kaldi::Fbank fbank(fbank_opts);

        // Compute Fbank features
        kaldi::Matrix<kaldi::BaseFloat> features;
        fbank.ComputeFeatures(mono_waveform, sample_rate, 1.0, &features);

        // cmvn
        kaldi::Matrix<double> cmvn_stats(2, features.NumCols() + 1);
        cmvn_stats.SetZero();
        AccCmvnStats(features, nullptr, &cmvn_stats);
        ApplyCmvn(cmvn_stats, false, &features);

        // Allocate memory for output
        *rows = features.NumRows();
        *cols = features.NumCols();
        *output = new float[(*rows) * (*cols)];

        // Copy features to output
        for (int i = 0; i < *rows; ++i) {
            for (int j = 0; j < *cols; ++j) {
                (*output)[i * (*cols) + j] = features(i, j);
            }
        }

        return 0;  // Success
    } catch (const std::exception& e) {
        std::cerr << "Error during Fbank computation: " << e.what() << std::endl;
        return -1;
    }
}

// Free allocated memory
void free_fbank_output(float* output) {
    delete[] output;
}

}
