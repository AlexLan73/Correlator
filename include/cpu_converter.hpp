#ifndef CPU_CONVERTER_HPP
#define CPU_CONVERTER_HPP

#include <cstdint>
#include <CL/cl.h>
#include <string>
#include "profiler.hpp"

/**
 * Параметры для GPU конвертации
 * (используется для унификации интерфейса)
 */
struct GPUConversionParams {
    size_t N;
    int num_shifts;
    int num_input_vectors;
    float scale_factor;
    
    size_t input_ref_size;      // байты
    size_t input_data_size;     // байты
    size_t output_ref_size;     // байты
    size_t output_data_size;    // байты
};

/**
 * Конвертировать int32 → float2 с циклическими сдвигами (опорные сигналы)
 * 
 * Использует OpenMP параллелизм для максимальной скорости на CPU.
 * Каждый thread обрабатывает один циклический сдвиг.
 * 
 * @param input        входные int32[N] данные
 * @param output       выходные float2[num_shifts × N] данные
 * @param N            размер входного вектора (например, 2^15)
 * @param num_shifts   количество циклических сдвигов (например, 40)
 * @param scale_factor масштабирование (например, 1/32768.0f для нормализации)
 * @param profiler     объект профилирования
 * @param profile_label метка для профилирования
 */
void convert_reference_signals_cpu(
    const int32_t* input,
    cl_float2* output,
    size_t N,
    int num_shifts,
    float scale_factor,
    Profiler& profiler,
    const std::string& profile_label
);

/**
 * Конвертировать int32 → float2 для входных данных
 * 
 * Простая конвертация без циклических сдвигов.
 * Используется максимальный OpenMP параллелизм.
 * 
 * @param input         входные int32[num_vectors × N] данные
 * @param output        выходные float2[num_vectors × N] данные
 * @param N             размер каждого вектора
 * @param num_vectors   количество векторов (например, 50)
 * @param scale_factor  масштабирование
 * @param profiler      объект профилирования
 * @param profile_label метка для профилирования
 */
void convert_input_signals_cpu(
    const int32_t* input,
    cl_float2* output,
    size_t N,
    int num_vectors,
    float scale_factor,
    Profiler& profiler,
    const std::string& profile_label
);

/**
 * Подготовить параметры для GPU конвертации
 */
void prepare_gpu_conversion_params(
    GPUConversionParams& params,
    size_t N,
    int num_shifts,
    int num_input_vectors,
    float scale_factor
);

/**
 * Валидировать correctness конвертации
 * (сравнить CPU результаты с GPU результатами)
 * 
 * @param cpu_result    результаты CPU конвертации
 * @param gpu_result    результаты GPU конвертации
 * @param num_elements  количество элементов для сравнения
 * @param tolerance     максимально допустимая ошибка
 * @return true если совпадают, false иначе
 */
bool validate_conversion(
    const cl_float2* cpu_result,
    const cl_float2* gpu_result,
    size_t num_elements,
    float tolerance = 1e-5f
);

/**
 * Benchmark: профилировать конвертацию для различных размеров N
 */
void benchmark_conversion(
    Profiler& profiler,
    int num_runs = 5
);

#endif // CPU_CONVERTER_HPP
