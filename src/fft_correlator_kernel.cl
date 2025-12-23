// ============================================================================
// FFT Correlator OpenCL Kernels
// Pre-Callback для конвертации int32 → float2 + циклические сдвиги
// ============================================================================

/**
 * Pre-Callback kernel для опорных сигналов (reference signals)
 * 
 * Задача:
 * 1. Конвертировать int32 → float2 (нормализованные комплексные числа)
 * 2. Применить циклические сдвиги (все 40 вариантов)
 * 
 * Параметры (в userdata как один длинный вектор uint32):
 *   params[0] = n_shifts (40)
 *   params[1] = fft_size (2^15 = 32768)
 *   params[2] = is_hamming (0/1)
 *   params[3] = scale_factor_exp (если нужна)
 * 
 * Формат данных:
 *   Вход:  int32[fft_size] → один опорный сигнал
 *   Выход: float2[n_shifts × fft_size] → все циклические сдвиги
 * 
 * Выполнение:
 *   - Каждый работный элемент обрабатывает одну позицию для одного сдвига
 *   - Смещение считывается как: src_idx = (idx + shift) % fft_size
 */
__kernel void prepare_reference_signals_pre_callback(
    __global float2* restrict output,      // Выход: complex[n_shifts × fft_size]
    const __global int* restrict userdata  // Параметры + входные данные
) {
    // Глобальный индекс нити
    size_t gid = get_global_id(0);
    
    // Прочитать параметры из userdata[0..3]
    uint n_shifts = userdata[0];           // 40
    uint fft_size = userdata[1];           // 2^15
    uint is_hamming = userdata[2];         // 0 или 1
    uint scale_factor_exp = userdata[3];   // экспонента
    
    // Выход имеет размер n_shifts × fft_size
    size_t total_elements = (size_t)n_shifts * fft_size;
    
    if (gid >= total_elements) return;
    
    // Разложить gid на индекс сдвига и позицию в сигнале
    uint shift_idx = gid / fft_size;
    uint pos_idx = gid % fft_size;
    
    // Смещение для этого конкретного сдвига
    uint cyclic_shift = shift_idx;  // 0, 1, 2, ..., 39
    
    // Вычислить исходный индекс с циклическим сдвигом
    uint src_idx = (pos_idx + cyclic_shift) % fft_size;
    
    // ========================================================================
    // ЧИТАЕМ ВХОДНЫЕ ДАННЫЕ
    // ========================================================================
    
    // Входные данные начинаются с userdata[4 + hamming_size]
    // Для простоты сначала предположим, что они идут после параметров
    // В реальном коде нужно передать смещение в userdata
    
    size_t input_offset = 4;  // После 4 параметров (в words, не bytes!)
    const __global int* input_signal = (const __global int*)(userdata + input_offset);
    
    int sample_int32 = input_signal[src_idx];
    
    // ========================================================================
    // КОНВЕРТАЦИЯ: int32 → float
    // ========================================================================
    
    // Нормализовать: int32 в диапазон [-1, 1]
    float scale = 1.0f / 32768.0f;  // или 2^15
    float real_part = (float)sample_int32 * scale;
    
    // ========================================================================
    // ПРИМЕНИТЬ ОКНО ХЕММИНГА (опционально)
    // ========================================================================
    
    float window = 1.0f;
    if (is_hamming) {
        // Hamming окно: 0.54 - 0.46 * cos(2π * n / (N-1))
        float n_norm = (float)pos_idx / (float)(fft_size - 1);
        window = 0.54f - 0.46f * cos(2.0f * M_PI_F * n_norm);
    }
    
    real_part *= window;
    
    // ========================================================================
    // ВЫХОД: complex число (real, imaginary)
    // ========================================================================
    
    output[gid] = (float2)(real_part, 0.0f);  // Чисто вещественный сигнал
}

/**
 * Pre-Callback kernel для входных сигналов (input signals)
 * 
 * Задача:
 * 1. Конвертировать int32 → float2 (50 входных сигналов параллельно)
 * 2. Нормализировать
 * 
 * Параметры (в userdata как один длинный вектор uint32):
 *   params[0] = n_signals (50)
 *   params[1] = fft_size (2^15)
 *   params[2] = is_hamming (0/1)
 *   params[3] = scale_factor_exp
 * 
 * Формат данных:
 *   Вход:  int32[n_signals × fft_size] → все входные сигналы
 *   Выход: float2[n_signals × fft_size] → конвертированные
 */
__kernel void prepare_input_signals_pre_callback(
    __global float2* restrict output,      // Выход: complex[n_signals × fft_size]
    const __global int* restrict userdata  // Параметры + входные данные
) {
    size_t gid = get_global_id(0);
    
    uint n_signals = userdata[0];    // 50
    uint fft_size = userdata[1];     // 2^15
    uint is_hamming = userdata[2];   // 0 или 1
    uint scale_factor_exp = userdata[3];
    
    size_t total_elements = (size_t)n_signals * fft_size;
    
    if (gid >= total_elements) return;
    
    // Разложить gid на индекс сигнала и позицию
    uint signal_idx = gid / fft_size;
    uint pos_idx = gid % fft_size;
    
    // Входные данные начинаются после параметров
    size_t input_offset = 4;  // После 4 параметров
    const __global int* input_signals = (const __global int*)(userdata + input_offset);
    
    // Прочитать значение для этого сигнала и позиции
    int sample_int32 = input_signals[gid];  // Линейное расположение
    
    // ========================================================================
    // КОНВЕРТАЦИЯ: int32 → float
    // ========================================================================
    
    float scale = 1.0f / 32768.0f;
    float real_part = (float)sample_int32 * scale;
    
    // ========================================================================
    // ПРИМЕНИТЬ ОКНО ХЕММИНГА (опционально)
    // ========================================================================
    
    float window = 1.0f;
    if (is_hamming) {
        float n_norm = (float)pos_idx / (float)(fft_size - 1);
        window = 0.54f - 0.46f * cos(2.0f * M_PI_F * n_norm);
    }
    
    real_part *= window;
    
    // ========================================================================
    // ВЫХОД: complex число (real, imaginary)
    // ========================================================================
    
    output[gid] = (float2)(real_part, 0.0f);
}

// ============================================================================
// Post-Callback kernel будет отдельно (для шага 3)
// ============================================================================
