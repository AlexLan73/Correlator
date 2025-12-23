#include "cpu_converter.hpp"
#include "profiler.hpp"
#include <omp.h>
#include <cstring>
#include <cmath>

/**
 * Конвертировать int32 → float2 с циклическими сдвигами для опорных сигналов
 * 
 * @param input           входные данные int32[N]
 * @param output          выходные данные float2[num_shifts × N]
 * @param N               размер входного вектора (2^15)
 * @param num_shifts      количество циклических сдвигов (40)
 * @param scale_factor    множитель масштабирования (для нормализации int32 → float)
 * @param profiler        объект для профилирования
 * @param profile_label   метка для профилирования
 * 
 * Особенности:
 * - Распараллеливается по сдвигам (внешний loop)
 * - Каждый поток работает с собственным буфером (cache locality)
 * - L1 cache оптимизация: N × 8 байт на поток
 */
void convert_reference_signals_cpu(
    const int32_t* input,
    cl_float2* output,
    size_t N,
    int num_shifts,
    float scale_factor,
    Profiler& profiler,
    const std::string& profile_label
) {
    if (!input || !output || N == 0) {
        fprintf(stderr, "ERROR: Invalid parameters in convert_reference_signals_cpu\n");
        return;
    }
    
    profiler.start(profile_label);
    
    // Параллелизм по сдвигам (внешний loop)
    // Каждый thread обрабатывает отдельный сдвиг → минимум синхронизации
    #pragma omp parallel for collapse(2) schedule(static)
    for (int shift = 0; shift < num_shifts; shift++) {
        for (int i = 0; i < static_cast<int>(N); i++) {
            // Индекс с циклическим сдвигом
            size_t input_idx = (i + shift) % N;
            
            // Индекс в выходном буфере
            size_t output_idx = shift * N + i;
            
            // Конвертация int32 → float2 (мнимая часть = 0)
            output[output_idx].s[0] = (float)input[input_idx] * scale_factor;
            output[output_idx].s[1] = 0.0f;  // Мнимая часть нулевая для опорных сигналов
        }
    }
    
    double elapsed_us = profiler.stop(profile_label, Profiler::MICROSECONDS);
    
    #ifdef VERBOSE_PROFILING
    printf("[CPU] convert_reference_signals: %.3f μs (N=%zu, shifts=%d, %.1f GB/s)\n",
           elapsed_us,
           N,
           num_shifts,
           (N * num_shifts * sizeof(int32_t) + N * num_shifts * sizeof(cl_float2)) / 
           (elapsed_us / 1e6) / 1e9);
    #endif
}

/**
 * Конвертировать int32 → float2 для входных данных
 * 
 * @param input           входные данные int32[num_vectors × N]
 * @param output          выходные данные float2[num_vectors × N]
 * @param N               размер каждого вектора (2^15)
 * @param num_vectors     количество векторов (50)
 * @param scale_factor    множитель масштабирования
 * @param profiler        объект для профилирования
 * @param profile_label   метка для профилирования
 * 
 * Особенности:
 * - Простая конвертация без циклических сдвигов
 * - Максимальная параллелизация (все элементы независимы)
 * - Оптимальное использование cache (линейный доступ к памяти)
 */
void convert_input_signals_cpu(
    const int32_t* input,
    cl_float2* output,
    size_t N,
    int num_vectors,
    float scale_factor,
    Profiler& profiler,
    const std::string& profile_label
) {
    if (!input || !output || N == 0) {
        fprintf(stderr, "ERROR: Invalid parameters in convert_input_signals_cpu\n");
        return;
    }
    
    profiler.start(profile_label);
    
    size_t total_elements = N * num_vectors;
    
    // Параллелизм по элементам (максимальная простота)
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < static_cast<int>(total_elements); idx++) {
        output[idx].s[0] = (float)input[idx] * scale_factor;
        output[idx].s[1] = 0.0f;  // Мнимая часть нулевая для вещественных входных данных
    }
    
    double elapsed_us = profiler.stop(profile_label, Profiler::MICROSECONDS);
    
    #ifdef VERBOSE_PROFILING
    printf("[CPU] convert_input_signals: %.3f μs (N=%zu, vectors=%d, %.1f GB/s)\n",
           elapsed_us,
           N,
           num_vectors,
           (total_elements * sizeof(int32_t) + total_elements * sizeof(cl_float2)) / 
           (elapsed_us / 1e6) / 1e9);
    #endif
}

/**
 * Версия для уже загруженных данных на GPU
 * (конвертация will happen on GPU side)
 * 
 * Этот файл используется для подготовки параметров профилирования
 */
void prepare_gpu_conversion_params(
    GPUConversionParams& params,
    size_t N,
    int num_shifts,
    int num_input_vectors,
    float scale_factor
) {
    params.N = N;
    params.num_shifts = num_shifts;
    params.num_input_vectors = num_input_vectors;
    params.scale_factor = scale_factor;
    
    // Размеры буферов
    params.input_ref_size = N * sizeof(int32_t);
    params.input_data_size = num_input_vectors * N * sizeof(int32_t);
    params.output_ref_size = num_shifts * N * sizeof(cl_float2);
    params.output_data_size = num_input_vectors * N * sizeof(cl_float2);
}

/**
 * Сравнить correctness CPU vs GPU конвертации
 * (используется для валидации)
 */
bool validate_conversion(
    const cl_float2* cpu_result,
    const cl_float2* gpu_result,
    size_t num_elements,
    float tolerance
) {
    size_t errors = 0;
    const size_t max_errors_to_report = 10;
    
    for (size_t i = 0; i < num_elements; i++) {
        float diff_x = fabs(cpu_result[i].s[0] - gpu_result[i].s[0]);
        float diff_y = fabs(cpu_result[i].s[1] - gpu_result[i].s[1]);
        
        if (diff_x > tolerance || diff_y > tolerance) {
            errors++;
            if (errors <= max_errors_to_report) {
                printf("ERROR at index %zu: CPU=(%f, %f) vs GPU=(%f, %f)\n",
                       i, cpu_result[i].s[0], cpu_result[i].s[1],
                       gpu_result[i].s[0], gpu_result[i].s[1]);
            }
        }
    }
    
    if (errors > 0) {
        printf("Validation FAILED: %zu/%zu elements differ (tolerance=%.2e)\n",
               errors, num_elements, tolerance);
        return false;
    } else {
        printf("Validation OK: All %zu elements match (tolerance=%.2e)\n",
               num_elements, tolerance);
        return true;
    }
}

/**
 * Benchmark конвертации: различные размеры N
 */
void benchmark_conversion(
    Profiler& profiler,
    int num_runs
) {
    printf("\n========== CONVERSION BENCHMARK ==========\n");
    printf("Running %d iterations per configuration...\n\n", num_runs);
    
    const std::vector<size_t> sizes = {
        (1 << 10),  // 1K
        (1 << 12),  // 4K
        (1 << 15),  // 32K (опорные)
        (1 << 16),  // 64K
        (1 << 18),  // 256K
    };
    
    for (size_t N : sizes) {
        std::vector<int32_t> input(N);
        std::vector<cl_float2> output(N);
        
        // Инициализация
        for (int i = 0; i < static_cast<int>(N); i++) {
            input[i] = static_cast<int32_t>(i % 1000);
        }
        
        // Прогрев
        for (int i = 0; i < 2; i++) {
            #pragma omp parallel for
            for (int j = 0; j < static_cast<int>(N); j++) {
                output[j].s[0] = (float)input[j] / 1000.0f;
                output[j].s[1] = 0.0f;
            }
        }
        
        // Собственно тест
        std::string label = "convert_" + std::to_string(N);
        for (int run = 0; run < num_runs; run++) {
            profiler.start(label);
            
            #pragma omp parallel for
            for (int i = 0; i < static_cast<int>(N); i++) {
                output[i].s[0] = (float)input[i] / 1000.0f;
                output[i].s[1] = 0.0f;
            }
            
            profiler.stop(label, Profiler::MICROSECONDS);
        }
        
        // Результаты для этого N
        double avg_us = profiler.get_avg(label);
        double throughput_gb_s = (N * sizeof(int32_t) + N * sizeof(cl_float2)) / 
                                  (avg_us / 1e6) / 1e9;
        
        printf("N=%7zu: %.3f μs (avg), %.2f GB/s throughput\n",
               N, avg_us, throughput_gb_s);
        
        profiler.clear(label);
    }
    
    printf("========================================\n\n");
}
