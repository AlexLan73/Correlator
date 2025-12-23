/**
 * GPU Kernels для конвертации данных
 * 
 * Файл: gpu_converter_kernel.cl
 * Язык: OpenCL C
 * 
 * Kernels:
 * 1. convert_int32_to_float2 — конвертация int32 → float2 (Re, Im=0)
 * 2. apply_cyclic_shifts — организация циклических сдвигов
 */

// ============================================================================
// KERNEL 1: Simple conversion int32 → float2
// ============================================================================

/**
 * Конвертировать int32 → float2 (simple version without shifts)
 * 
 * @param input      входной буфер int32[num_elements]
 * @param output     выходной буфер float2[num_elements]
 * @param scale      множитель масштабирования (например, 1/32768.0f)
 * @param num_elements общее количество элементов
 * 
 * Использование:
 * - Один work-item на один элемент
 * - Линейный доступ к памяти (coalesced access)
 * - Минимум синхронизации
 * 
 * Performance:
 * - Bandwidth limited (compute intensity = 0.25)
 * - Лучше запускать вместе с другими операциями
 */
__kernel void convert_int32_to_float2(
    __global const int* input,
    __global float2* output,
    const float scale,
    const unsigned int num_elements
) {
    unsigned int gid = get_global_id(0);
    
    // Граница: каждый thread обрабатывает один элемент
    if (gid >= num_elements) return;
    
    // Чтение (4 bytes)
    int val = input[gid];
    
    // Конвертация: int32 → float2 (Im = 0)
    float2 result = (float2)((float)val * scale, 0.0f);
    
    // Запись (8 bytes)
    output[gid] = result;
}

// ============================================================================
// KERNEL 2: Conversion with cyclic shifts
// ============================================================================

/**
 * Конвертировать int32 → float2 с циклическими сдвигами
 * 
 * @param input         входной буфер int32[N]
 * @param output        выходной буфер float2[num_shifts * N]
 * @param scale         множитель масштабирования
 * @param N             размер входного вектора (2^15)
 * @param num_shifts    количество сдвигов (40)
 * 
 * Организация памяти в выходном буфере:
 * ┌─────────────────────────────────┐
 * │ shift=0: [float2_0, float2_1, ...] (N элементов)
 * │ shift=1: [float2_N, float2_{N+1}, ...] (N элементов)
 * │ ...
 * │ shift=39: [float2_{39N}, ...] (N элементов)
 * └─────────────────────────────────┘
 * 
 * Использование:
 * - Один work-item на один элемент выходного буфера
 * - Работает с циклической индексацией: (i + shift) % N
 * - Write-amplification: читаем 4 bytes, пишем 8 bytes
 * 
 * Performance:
 * - Compute intensity: 1 операция на 8 bytes output
 * - Memory bound, но нормально для GPU
 */
__kernel void apply_cyclic_shifts(
    __global const int* input,
    __global float2* output,
    const float scale,
    const unsigned int N,
    const unsigned int num_shifts
) {
    unsigned int gid = get_global_id(0);
    unsigned int total_output_size = N * num_shifts;
    
    if (gid >= total_output_size) return;
    
    // Декодировать глобальный индекс в (shift, position)
    unsigned int shift = gid / N;        // какой сдвиг
    unsigned int position = gid % N;     // позиция внутри сдвига
    
    // Циклический сдвиг: (position + shift) % N
    unsigned int input_idx = (position + shift) % N;
    
    // Чтение с циклическим сдвигом
    int val = input[input_idx];
    
    // Конвертация и запись
    output[gid] = (float2)((float)val * scale, 0.0f);
}

// ============================================================================
// KERNEL 3: Batch version (multiple shifts in one kernel call)
// ============================================================================

/**
 * Оптимизированная версия для батч обработки
 * (если нужна лучшая локальность)
 * 
 * @param input         входной буфер int32[N]
 * @param output        выходной буфер float2[num_shifts * N]
 * @param scale         множитель масштабирования
 * @param N             размер входного вектора
 * @param num_shifts    количество сдвигов
 * @param shift_start   начальный сдвиг (для разделения работы)
 * @param num_shifts_to_process количество сдвигов для обработки в этом вызове
 * 
 * Используется для параллельной обработки нескольких сдвигов
 * разными command queues или разными kernel invocations
 */
__kernel void apply_cyclic_shifts_batch(
    __global const int* input,
    __global float2* output,
    const float scale,
    const unsigned int N,
    const unsigned int shift_start,
    const unsigned int num_shifts_to_process
) {
    unsigned int gid = get_global_id(0);
    unsigned int total_elements = N * num_shifts_to_process;
    
    if (gid >= total_elements) return;
    
    // Декодировать индекс внутри батча
    unsigned int shift_in_batch = gid / N;
    unsigned int position = gid % N;
    
    // Абсолютный сдвиг
    unsigned int shift = shift_start + shift_in_batch;
    
    // Циклический индекс в исходном буфере
    unsigned int input_idx = (position + shift) % N;
    
    // Чтение и конвертация
    int val = input[input_idx];
    
    // Вычислить выходной индекс (глобальный, не локальный)
    unsigned int output_idx = (shift * N) + position;
    
    output[output_idx] = (float2)((float)val * scale, 0.0f);
}

// ============================================================================
// KERNEL 4: Optimized with local memory (if available)
// ============================================================================

/**
 * Оптимизированная версия с локальной памятью
 * Используется для уменьшения глобальных обращений (если нужна скорость)
 * 
 * Требует tuning for specific GPU (LDS size, work group size)
 */
__kernel void apply_cyclic_shifts_optimized(
    __global const int* input,
    __global float2* output,
    __local float2* local_cache,        // LDS буфер размером work_group_size * 2
    const float scale,
    const unsigned int N,
    const unsigned int num_shifts
) {
    unsigned int gid = get_global_id(0);
    unsigned int lid = get_local_id(0);
    unsigned int lsize = get_local_size(0);
    unsigned int total_output_size = N * num_shifts;
    
    if (gid >= total_output_size) return;
    
    // Декодировать индекс
    unsigned int shift = gid / N;
    unsigned int position = gid % N;
    unsigned int input_idx = (position + shift) % N;
    
    // Чтение из глобальной памяти
    int val = input[input_idx];
    
    // Конвертация в локальную память (для коалесцирования)
    float2 result = (float2)((float)val * scale, 0.0f);
    local_cache[lid] = result;
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Запись в глобальную память
    output[gid] = local_cache[lid];
}

// ============================================================================
// UTILITY KERNEL: Fill buffer (для тестов)
// ============================================================================

/**
 * Заполнить буфер тестовыми данными
 * (используется для бенчмарков)
 */
__kernel void fill_test_data(
    __global int* output,
    const unsigned int num_elements,
    const int seed
) {
    unsigned int gid = get_global_id(0);
    if (gid >= num_elements) return;
    
    // Простой LCG генератор для воспроизводимости
    unsigned int val = (seed + gid * 123) % 10000;
    output[gid] = (int)val - 5000;
}

/**
 * Verify correctness: сравнить CPU и GPU результаты
 */
__kernel void verify_results(
    __global const float2* cpu_result,
    __global const float2* gpu_result,
    __global int* errors,
    const unsigned int num_elements,
    const float tolerance
) {
    unsigned int gid = get_global_id(0);
    if (gid >= num_elements) return;
    
    float2 cpu = cpu_result[gid];
    float2 gpu = gpu_result[gid];
    
    float diff_x = fabs(cpu.x - gpu.x);
    float diff_y = fabs(cpu.y - gpu.y);
    
    if (diff_x > tolerance || diff_y > tolerance) {
        atomic_inc(&errors[0]);
    }
}
