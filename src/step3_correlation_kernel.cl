#include "fft_handler.hpp"
#include <cstdio>
#include <cstring>

// ============================================================================
// STEP 3: Correlation (Multiply + IFFT + Post-callback)
// ============================================================================

/**
 * Complex multiply: C = A * conj(B)
 * Used for correlation in frequency domain
 */
__kernel void complex_multiply_kernel(
    __global const float2* reference_fft,   // Reference spectrum (40 × 32768)
    __global const float2* input_fft,       // Input spectrum (50 × 32768)
    __global float2* correlation_fft,       // Output (50 × 40 × 32768)
    uint num_shifts,                        // 40
    uint num_signals,                       // 50
    uint fft_size                           // 32768
) {
    uint gid = get_global_id(0);
    uint total_elements = num_signals * num_shifts * fft_size;
    
    if (gid >= total_elements) return;
    
    // Decompose gid into: signal_idx, shift_idx, element_idx
    uint element_idx = gid % fft_size;
    uint shift_idx = (gid / fft_size) % num_shifts;
    uint signal_idx = gid / (fft_size * num_shifts);
    
    // Get reference and input spectrum elements
    float2 ref = reference_fft[shift_idx * fft_size + element_idx];
    float2 inp = input_fft[signal_idx * fft_size + element_idx];
    
    // Complex multiply: result = ref * conj(inp)
    // (a + bi) * (c - di) = (ac + bd) + (bc - ad)i
    float real_part = ref.x * inp.x + ref.y * inp.y;
    float imag_part = ref.y * inp.x - ref.x * inp.y;
    
    correlation_fft[gid] = (float2)(real_part, imag_part);
}

// ============================================================================
// Post-Callback Kernel: Find peaks and extract N_KG points
// ============================================================================

__kernel void post_callback_find_peaks(
    __global const float2* ifft_result,     // IFFT results (time domain)
    __global float* peaks_output,           // Output peaks
    uint num_signals,                       // 50
    uint num_shifts,                        // 40
    uint fft_size,                          // 32768
    uint n_kg,                              // 5 (num output points)
    uint peak_search_range                  // FFT_SIZE/2
) {
    uint gid = get_global_id(0);
    uint total_correlations = num_signals * num_shifts;
    
    if (gid >= total_correlations) return;
    
    uint signal_idx = gid / num_shifts;
    uint shift_idx = gid % num_shifts;
    
    // Find maximum in this correlation
    float max_val = 0.0f;
    uint max_idx = 0;
    
    uint base_idx = gid * fft_size;
    
    for (uint i = 0; i < peak_search_range; i++) {
        float magnitude = sqrt(
            ifft_result[base_idx + i].x * ifft_result[base_idx + i].x +
            ifft_result[base_idx + i].y * ifft_result[base_idx + i].y
        );
        
        if (magnitude > max_val) {
            max_val = magnitude;
            max_idx = i;
        }
    }
    
    // Extract top N_KG points (for now just output max)
    uint output_idx = (signal_idx * num_shifts + shift_idx) * n_kg;
    
    for (uint k = 0; k < n_kg; k++) {
        if (k == 0) {
            peaks_output[output_idx + k] = max_val;
        } else {
            peaks_output[output_idx + k] = 0.0f;
        }
    }
}
