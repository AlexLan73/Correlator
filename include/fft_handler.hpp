#ifndef FFT_HANDLER_HPP
#define FFT_HANDLER_HPP

#include <CL/opencl.h>
#include <clFFT.h>
#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>
#include "debug_log.hpp"

// ============================================================================
// FFT Handler для коррелятора
// ============================================================================

/**
 * Структура для хранения GPU контекста и FFT плана
 */
struct FFTContext {
    cl_context context;
    cl_command_queue queue;
    cl_device_id device;
    
    clfftPlanHandle reference_fft_plan;   // План для опорных сигналов (40 FFT)
    clfftPlanHandle input_fft_plan;       // План для входных сигналов (50 FFT)
    clfftPlanHandle correlation_ifft_plan; // План для IFFT корреляции (2000 IFFT)
    
    cl_mem reference_data;     // Входные int32 опорные сигналы
    cl_mem reference_fft;      // Выходные complex спектры опорных (40 × N)
    
    cl_mem input_data;         // Входные int32 входные сигналы (50 × N)
    cl_mem input_fft;          // Выходные complex спектры входных (50 × N)
    
    cl_mem correlation_fft;    // Промежуточные спектры корреляции (2000 × N)
    cl_mem correlation_ifft;   // Результаты IFFT (2000 × N)
    
    cl_mem pre_callback_userdata;   // Userdata для pre-callback (Step 1, 2)
    cl_mem pre_callback_userdata_correlation; // Userdata для pre-callback Complex Multiply (Step 3)
    cl_mem post_callback_userdata;  // Userdata для post-callback
    
    bool initialized;
    bool is_cleaned_up;  //флаг очистки

    FFTContext() 
        : context(nullptr), queue(nullptr), device(nullptr),
          reference_fft_plan(0), input_fft_plan(0), correlation_ifft_plan(0),
          reference_data(nullptr), reference_fft(nullptr),
          input_data(nullptr), input_fft(nullptr),
          correlation_fft(nullptr), correlation_ifft(nullptr),
          pre_callback_userdata(nullptr), pre_callback_userdata_correlation(nullptr), post_callback_userdata(nullptr),
          initialized(false), is_cleaned_up(false) {}
};

/**
 * Параметры для Pre-Callback (конвертация int32 → float2 + циклические сдвиги)
 * Передаём как один длинный вектор uint32
 */
struct PreCallbackParams {
    cl_uint n_shifts;           // [0] количество циклических сдвигов (40)
    cl_uint fft_size;           // [1] размер FFT (2^15)
    cl_uint is_hamming;         // [2] флаг окна Хемминга (0/1)
    cl_uint scale_factor_exp;   // [3] экспонента для масштабирования (если нужна)
    
    std::vector<cl_uint> to_vector() const {
        return {n_shifts, fft_size, is_hamming, scale_factor_exp};
    }
};

/**
 * Параметры для Post-Callback (выборка пиков и выходные данные)
 * Передаём как один длинный вектор uint32
 */
struct PostCallbackParams {
    cl_uint n_signals;          // [0] количество входных сигналов (50)
    cl_uint n_correlators;      // [1] количество опорных/сдвигов (40)
    cl_uint fft_size;           // [2] размер FFT
    cl_uint n_kg;               // [3] кол-во выводимых точек (параметр)
    cl_uint peak_search_range;  // [4] диапазон поиска пика (обычно fft_size/2)
    
    std::vector<cl_uint> to_vector() const {
        return {n_signals, n_correlators, fft_size, n_kg, peak_search_range};
    }
};

/**
 * Параметры для Pre-Callback Complex Multiply (Step 3)
 * Используется в IFFT плане для перемножения спектров
 */
struct ComplexMultiplyPreCallbackParams {
    cl_uint num_signals;        // [0] количество входных сигналов (50)
    cl_uint num_shifts;         // [1] количество сдвигов (40)
    cl_uint fft_size;           // [2] размер FFT (32768)
    cl_uint padding;            // [3] выравнивание
    // Note: указатели на reference_fft и input_fft передаются через
    // глобальные буферы, доступные через контекст OpenCL
    
    std::vector<cl_uint> to_vector() const {
        return {num_signals, num_shifts, fft_size, padding};
    }
};

// ============================================================================
// FFT Handler Class
// ============================================================================

class FFTHandler {
public:
    /**
     * Конструктор
     */
    FFTHandler(cl_context ctx, cl_command_queue q, cl_device_id dev) {

        if (!ctx || !q || !dev) {
            throw std::runtime_error("Invalid OpenCL context/queue/device");
        }

        // Инициализация контекста
        ctx_.context = ctx;
        ctx_.queue = q;
        ctx_.device = dev;
        ctx_.reference_fft_plan = 0;
        ctx_.input_fft_plan = 0;
        ctx_.correlation_ifft_plan = 0;
        ctx_.reference_data = nullptr;
        ctx_.reference_fft = nullptr;
        ctx_.input_data = nullptr;
        ctx_.input_fft = nullptr;
        ctx_.correlation_fft = nullptr;
        ctx_.correlation_ifft = nullptr;
        ctx_.pre_callback_userdata = nullptr;
        ctx_.post_callback_userdata = nullptr;
        ctx_.initialized = false;
        
        // Инициализация сохраненных параметров
        fft_size_ = 0;
        num_shifts_ = 0;
        num_signals_ = 0;
        n_kg_ = 0;
        scale_factor_ = 0.0f;
    }
    
    /**
     * Инициализировать FFT handler (создать буферы и планы)
     */
    void initialize(
        size_t N,                      // размер сигнала (2^15)
        int num_shifts,                // кол-во циклических сдвигов (40)
        int num_signals,               // кол-во входных сигналов (50)
        int n_kg,                      // кол-во выводимых точек (5)
        float scale_factor             // масштабирование для int32→float2
    );
    
    /**
     * Структура с детальными временами операции
     */
    struct OperationTiming {
        double execute_ms = 0.0;      // Время выполнения (START to END)
        double queue_wait_ms = 0.0;    // Ожидание в очереди (SUBMIT to START)
        double cpu_wait_ms = 0.0;      // CPU wait time (clWaitForEvents)
        double total_gpu_ms = 0.0;     // Общее GPU время (QUEUED to END)
    };
    
    /**
     * ШАГ 1: Загрузить опорный сигнал и запустить Forward FFT с pre-callback
     */
    void step1_reference_signals(
        const int32_t* host_reference,  // входной опорный сигнал
        size_t N,
        int num_shifts,
        float scale_factor,
        double& time_upload_ms,
        double& time_callback_ms,
        double& time_fft_ms,
        OperationTiming& upload_timing,
        OperationTiming& fft_timing
    );
    
    /**
     * ШАГ 2: Загрузить входные сигналы и запустить Forward FFT с pre-callback
     */
    void step2_input_signals(
        const int32_t* host_input,     // входные сигналы (num_signals × N)
        size_t N,
        int num_signals,
        float scale_factor,
        double& time_upload_ms,
        double& time_callback_ms,
        double& time_fft_ms,
        OperationTiming& upload_timing,
        OperationTiming& fft_timing
    );
    
    /**
     * ШАГ 3: Запустить корреляцию (multiplication + IFFT + post-callback)
     */
    void step3_correlation(
        int num_signals,
        int num_shifts,
        size_t N,
        int n_kg,
        double& time_multiply_ms,
        double& time_ifft_ms,
        double& time_download_ms,
        double& time_post_callback_ms,
        OperationTiming& multiply_timing,
        OperationTiming& ifft_timing,
        OperationTiming& download_timing
    );
    
    /**
     * Получить результаты корреляции
     * Формат: [num_signals][num_shifts][n_kg]
     */
    std::vector<std::vector<std::vector<float>>> get_correlation_results(
        int num_signals,
        int num_shifts,
        int n_kg
    );
    
    /**
     * Получить reference FFT данные (для валидации)
     * @param output Выходной вектор комплексных чисел
     * @param num_shifts Количество сдвигов
     * @param fft_size Размер FFT
     */
    bool getReferenceFFTData(std::vector<cl_float2>& output, int num_shifts, size_t fft_size) const;
    
    /**
     * Получить input FFT данные (для валидации)
     * @param output Выходной вектор комплексных чисел
     * @param num_signals Количество сигналов
     * @param fft_size Размер FFT
     */
    bool getInputFFTData(std::vector<cl_float2>& output, int num_signals, size_t fft_size) const;
    
    /**
     * Получить correlation peaks данные (для валидации)
     * @param output Выходной вектор пиков
     * @param num_signals Количество сигналов
     * @param num_shifts Количество сдвигов
     * @param n_kg Количество выходных точек
     */
    bool getCorrelationPeaksData(std::vector<float>& output, int num_signals, int num_shifts, int n_kg) const;
    
    /**
     * Получить размер FFT
     */
    size_t getFFTSize() const { return ctx_.initialized ? fft_size_ : 0; }
    
    /**
     * Освободить ресурсы
     */
    void cleanup();
    
    /**
     * Деструктор
     */
    ~FFTHandler() {
        cleanup();
    }

private:
    FFTContext ctx_;
    
    // Сохраненные параметры для доступа
    size_t fft_size_;
    int num_shifts_;
    int num_signals_;
    int n_kg_;
    float scale_factor_;
    
    /**
     * Создать 1D FFT план для батча
     */
    clfftPlanHandle create_fft_plan_1d(
        size_t fft_size,
        int batch_size,
        const std::string& plan_name
    );
    
    /**
     * Создать 1D FFT план с встроенным pre-callback
     */
    clfftPlanHandle create_fft_plan_1d_with_precallback(
        size_t fft_size,
        int batch_size,
        float scale_factor,
        const std::string& plan_name
    );
    
    /**
     * Создать 1D FFT план с встроенным pre-callback (int32→float2) и post-callback (complex conjugate)
     * Для Step 1: после Forward FFT выполняется комплексное сопряжение
     */
    clfftPlanHandle create_fft_plan_1d_with_pre_and_post_callback_conjugate(
        size_t fft_size,
        int batch_size,
        float scale_factor,
        const std::string& plan_name
    );
    
    /**
     * Создать 1D FFT план с встроенным post-callback
     */
    clfftPlanHandle create_fft_plan_1d_with_postcallback(
        size_t fft_size,
        int batch_size,
        int num_signals,
        int num_shifts,
        int n_kg,
        const std::string& plan_name
    );
    
    /**
     * Создать 1D IFFT план с встроенным pre-callback (Complex Multiply) и post-callback (Find Peaks)
     * Для Step 3 корреляции
     */
    clfftPlanHandle create_fft_plan_1d_with_pre_and_post_callback(
        size_t fft_size,
        int batch_size,
        int num_signals,
        int num_shifts,
        int n_kg,
        const std::string& plan_name
    );
    
    /**
     * Создать Pre-Callback userdata буфер
     */
    void create_pre_callback_userdata(
        size_t N,
        int num_shifts,
        const PreCallbackParams& params,
        const float* hamming_window = nullptr
    );
    
    /**
     * Создать Post-Callback userdata буфер
     */
    void create_post_callback_userdata(
        size_t N,
        int num_signals,
        int num_shifts,
        int n_kg,
        const PostCallbackParams& params
    );
    
    /**
     * Профилировать OpenCL событие
     */
    double profile_event(cl_event event, const std::string& label);
};

#endif // FFT_HANDLER_HPP
