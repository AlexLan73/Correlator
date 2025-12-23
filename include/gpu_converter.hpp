#ifndef GPU_CONVERTER_HPP
#define GPU_CONVERTER_HPP

#include <CL/cl.h>
#include <string>
#include <vector>
#include "profiler.hpp"

/**
 * Параметры GPU конвертации
 */
struct GPUConverterContext {
    cl_context context;
    cl_device_id device;
    cl_command_queue queue;
    
    // Скомпилированные kernels
    cl_kernel kernel_convert_simple;
    cl_kernel kernel_cyclic_shifts;
    cl_kernel kernel_cyclic_shifts_batch;
    cl_kernel kernel_fill_test_data;
    
    // Для профилирования
    cl_command_queue profiling_queue;  // с CL_QUEUE_PROFILING_ENABLE
};

/**
 * Инициализировать GPU контекст
 * 
 * @param ctx структура для заполнения
 * @param device_type CL_DEVICE_TYPE_GPU или CL_DEVICE_TYPE_CPU
 * @return CL_SUCCESS если успешно, иначе код ошибки
 */
cl_int init_gpu_context(
    GPUConverterContext& ctx,
    cl_device_type device_type = CL_DEVICE_TYPE_GPU
);

/**
 * Загрузить и скомпилировать OpenCL kernels из файла
 * 
 * @param ctx GPU контекст
 * @param kernel_file путь к .cl файлу
 * @return CL_SUCCESS если успешно
 */
cl_int load_kernels(
    GPUConverterContext& ctx,
    const std::string& kernel_file
);

/**
 * Очистить GPU контекст
 */
void cleanup_gpu_context(GPUConverterContext& ctx);

// ============================================================================
// GPU Conversion Functions
// ============================================================================

/**
 * Конвертировать int32 → float2 на GPU (простая версия без сдвигов)
 * 
 * @param ctx GPU контекст
 * @param d_input GPU буфер входных int32 данных
 * @param d_output GPU буфер выходных float2 данных
 * @param num_elements количество элементов
 * @param scale_factor масштабирование
 * @param profiler объект профилирования
 * @param profile_label метка для профилирования
 * @param event[out] OpenCL событие (для синхронизации)
 * @return CL_SUCCESS если успешно
 */
cl_int gpu_convert_simple(
    GPUConverterContext& ctx,
    cl_mem d_input,
    cl_mem d_output,
    size_t num_elements,
    float scale_factor,
    Profiler& profiler,
    const std::string& profile_label,
    cl_event* event = nullptr
);

/**
 * Конвертировать int32 → float2 с циклическими сдвигами на GPU
 * 
 * @param ctx GPU контекст
 * @param d_input GPU буфер входных int32 данных [N]
 * @param d_output GPU буфер выходных float2 данных [num_shifts * N]
 * @param N размер входного вектора
 * @param num_shifts количество циклических сдвигов
 * @param scale_factor масштабирование
 * @param profiler объект профилирования
 * @param profile_label метка для профилирования
 * @param event[out] OpenCL событие
 * @return CL_SUCCESS если успешно
 */
cl_int gpu_convert_cyclic_shifts(
    GPUConverterContext& ctx,
    cl_mem d_input,
    cl_mem d_output,
    size_t N,
    int num_shifts,
    float scale_factor,
    Profiler& profiler,
    const std::string& profile_label,
    cl_event* event = nullptr
);

/**
 * Батч версия: обработать несколько сдвигов параллельно
 * (может быть несколько вызовов, каждый для подмножества сдвигов)
 * 
 * @param ctx GPU контекст
 * @param d_input GPU буфер входных int32 данных [N]
 * @param d_output GPU буфер выходных float2 данных
 * @param N размер входного вектора
 * @param shift_start начальный сдвиг
 * @param num_shifts_to_process количество сдвигов в этом вызове
 * @param scale_factor масштабирование
 * @param profiler объект профилирования
 * @param profile_label метка для профилирования
 * @param event[out] OpenCL событие
 * @return CL_SUCCESS если успешно
 */
cl_int gpu_convert_cyclic_shifts_batch(
    GPUConverterContext& ctx,
    cl_mem d_input,
    cl_mem d_output,
    size_t N,
    int shift_start,
    int num_shifts_to_process,
    float scale_factor,
    Profiler& profiler,
    const std::string& profile_label,
    cl_event* event = nullptr
);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Заполнить GPU буфер тестовыми данными
 */
cl_int gpu_fill_test_data(
    GPUConverterContext& ctx,
    cl_mem d_output,
    size_t num_elements,
    int seed = 42
);

/**
 * Скопировать данные из CPU в GPU буфер
 */
cl_int gpu_copy_to_device(
    GPUConverterContext& ctx,
    const void* host_data,
    cl_mem device_buffer,
    size_t size,
    cl_event* event = nullptr
);

/**
 * Скопировать данные из GPU буфера в CPU
 */
cl_int gpu_copy_from_device(
    GPUConverterContext& ctx,
    cl_mem device_buffer,
    void* host_data,
    size_t size,
    cl_event* event = nullptr
);

/**
 * Получить информацию о GPU
 */
void print_gpu_info(const GPUConverterContext& ctx);

/**
 * Получить максимальный размер work group для kernel
 */
size_t get_max_work_group_size(
    GPUConverterContext& ctx,
    cl_kernel kernel
);

/**
 * Benchmark GPU конвертации
 */
void benchmark_gpu_conversion(
    GPUConverterContext& ctx,
    Profiler& profiler,
    int num_runs = 5
);

#endif // GPU_CONVERTER_HPP
