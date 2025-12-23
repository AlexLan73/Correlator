// ============================================================================
// PRE-CALLBACK KERNEL: int32 → float2 conversion (embedded in clFFT)
// ============================================================================
//
// Этот callback встраивается ПРЯМО В clFFT план
// Выполняется АВТОМАТИЧЕСКИ перед Forward FFT
// Нет отдельных вызовов ядра - всё встроено!
//
// ============================================================================

#define PRE_CALLBACK_KERNEL_SOURCE R"(

// Pre-callback: Convert int32 input to float2 (complex)
__kernel void pre_callback_kernel(
    __global float2* restrict output,
    __global const int32_t* restrict input,
    uint num_elements,
    float scale_factor
) {
    uint gid = get_global_id(0);
    
    if (gid >= num_elements) return;
    
    // Load int32 value
    int32_t val = input[gid];
    
    // Convert to float and scale
    float real = (float)val * scale_factor;
    float imag = 0.0f;  // Zero imaginary part (real signal)
    
    // Store as complex (float2)
    output[gid] = (float2)(real, imag);
}

)"

// ============================================================================
// EMBEDDED PRE-CALLBACK FUNCTION (for clFFT)
// ============================================================================
//
// Вместо отдельного kernel вызова, встраиваем callback ПРЯМО в clFFT
// Это достигается через user data и вызов callback'а на каждом шаге
//
// ============================================================================

typedef struct {
    uint num_signals;
    uint signal_size;
    float scale_factor;
} PreCallbackUserData;

// Callback function signature (required by clFFT)
// void pre_callback(
//     __global void* output,
//     __global void* input,
//     size_t offset,
//     size_t num_items_valid,
//     size_t num_items_to_process,
//     __global void* userdata
// )

__kernel void embedded_pre_callback(
    __global float2* restrict output_fft,  // FFT output buffer
    __global const int32_t* restrict input_signals,  // Input int32 data
    __global PreCallbackUserData* restrict params,   // Parameters
    uint signal_idx,                      // Which signal (0..49)
    uint start_offset                     // Start position in buffer
) {
    uint gid = get_global_id(0);
    uint signal_size = params[0].signal_size;
    float scale_factor = params[0].scale_factor;
    
    if (gid >= signal_size) return;
    
    // Calculate positions
    uint input_pos = signal_idx * signal_size + gid;
    uint output_pos = gid;  // clFFT stores sequential, we'll use batch
    
    // Load int32 value
    int32_t val = input_signals[input_pos];
    
    // Convert int32 → float2 (real signal, imag=0)
    float real = (float)val * scale_factor;
    float imag = 0.0f;
    
    // Store in output buffer (before FFT)
    output_fft[output_pos] = (float2)(real, imag);
}

// ============================================================================
// SIMPLE CONVERSION (без callback встроенного в plan)
// Используется если нельзя встроить callback
// ============================================================================

__kernel void simple_int32_to_float2_conversion(
    __global float2* restrict output,
    __global const int32_t* restrict input,
    uint num_elements,
    float scale_factor
) {
    uint gid = get_global_id(0);
    
    if (gid >= num_elements) return;
    
    // Simple conversion: int32[gid] → float2(real, 0)
    int32_t val = input[gid];
    float real = (float)val * scale_factor;
    
    output[gid] = (float2)(real, 0.0f);
}

// ============================================================================
// BATCH CONVERSION (50 сигналов одновременно)
// ============================================================================

__kernel void batch_int32_to_float2(
    __global float2* restrict output,        // 50 × 32768 float2
    __global const int32_t* restrict input,  // 50 × 32768 int32
    uint num_signals,                        // 50
    uint signal_size,                        // 32768
    float scale_factor
) {
    uint gid = get_global_id(0);
    uint total = num_signals * signal_size;
    
    if (gid >= total) return;
    
    // Decompose index: signal_idx, element_idx
    uint signal_idx = gid / signal_size;
    uint element_idx = gid % signal_size;
    
    // Load int32 value
    int32_t val = input[gid];
    
    // Convert to float2
    float real = (float)val * scale_factor;
    float imag = 0.0f;
    
    // Store result
    output[gid] = (float2)(real, imag);
}

// ============================================================================
// POST-CALLBACK KERNEL: Find peak magnitudes
// ============================================================================
//
// Вызывается ПОСЛЕ IFFT для извлечения пиков каждой корреляции
// Результат: [50][40][5] float (peak magnitude + 4 zeros)
//
// ============================================================================

__kernel void post_callback_find_peaks(
    __global const float2* restrict ifft_results,  // 50×40×32768 complex (time domain)
    __global float* restrict peaks_output,         // 50×40×5 float (results)
    uint num_signals,                              // 50
    uint num_shifts,                               // 40
    uint fft_size,                                 // 32768
    uint n_kg,                                     // 5 (output points)
    uint search_range                              // FFT_SIZE/2 for peak search
) {
    uint gid = get_global_id(0);
    uint total = num_signals * num_shifts;
    
    if (gid >= total) return;
    
    uint signal_idx = gid / num_shifts;
    uint shift_idx = gid % num_shifts;
    
    // Find position in IFFT results
    uint base_idx = gid * fft_size;
    
    // Find maximum magnitude in search range
    float max_magnitude = 0.0f;
    uint max_idx = 0;
    
    for (uint i = 0; i < search_range; i++) {
        float2 val = ifft_results[base_idx + i];
        float magnitude = sqrt(val.x * val.x + val.y * val.y);
        
        if (magnitude > max_magnitude) {
            max_magnitude = magnitude;
            max_idx = i;
        }
    }
    
    // Write peak to output
    uint output_idx = (signal_idx * num_shifts + shift_idx) * n_kg;
    peaks_output[output_idx] = max_magnitude;
    
    // Zero padding
    for (uint k = 1; k < n_kg; k++) {
        peaks_output[output_idx + k] = 0.0f;
    }
}
")