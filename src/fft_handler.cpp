#include "fft_handler.hpp"
#include "debug_log.hpp"
#include <cstdio>
#include <cmath>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>

// ============================================================================
// Helper: Load OpenCL kernel source from file
// ============================================================================

static std::string load_kernel_source(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open kernel file: " + filename);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// ============================================================================
// Helper: Profile OpenCL event with detailed timing
// ============================================================================

struct EventTiming {
    double queued_ms = 0.0;      // Time from enqueue to queued
    double submit_ms = 0.0;      // Time from queued to submit
    double queue_wait_ms = 0.0;  // Submit to start (queue wait time)
    double execute_ms = 0.0;     // Start to end (execution time)
    double total_ms = 0.0;       // Queued to end (total time)
    double wait_ms = 0.0;        // CPU wait time (clWaitForEvents)
};

EventTiming profile_event_detailed(cl_event event) {
    EventTiming timing;
    if (!event) return timing;
    
    auto wait_start = std::chrono::high_resolution_clock::now();
    cl_int err = clWaitForEvents(1, &event);
    auto wait_end = std::chrono::high_resolution_clock::now();
    
    if (err != CL_SUCCESS) return timing;
    
    // Measure CPU wait time
    timing.wait_ms = std::chrono::duration_cast<std::chrono::microseconds>(
        wait_end - wait_start).count() / 1000.0;
    
    cl_ulong time_queued = 0, time_submit = 0, time_start = 0, time_end = 0;
    
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(time_queued), &time_queued, nullptr);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, sizeof(time_submit), &time_submit, nullptr);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, nullptr);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, nullptr);
    
    // Convert from nanoseconds to milliseconds
    timing.queued_ms = (time_submit - time_queued) / 1e6;      // Queued to submit
    timing.queue_wait_ms = (time_start - time_submit) / 1e6;   // Submit to start
    timing.execute_ms = (time_end - time_start) / 1e6;         // Start to end
    timing.total_ms = (time_end - time_queued) / 1e6;          // Total GPU time
    timing.submit_ms = timing.queued_ms;  // Alias for clarity
    
    return timing;
}

double FFTHandler::profile_event(cl_event event, const std::string& label) {
    EventTiming timing = profile_event_detailed(event);
    double elapsed_ms = timing.execute_ms;
    DEBUG_LOG("  [PROFILE] %s: %.3f ms\n", label.c_str(), elapsed_ms);
    return elapsed_ms;
}

// ============================================================================
// Initialize FFT Handler
// ============================================================================

void FFTHandler::initialize(
    size_t N,
    int num_shifts,
    int num_signals,
    int n_kg,
    float scale_factor
) {
    if (ctx_.initialized) {
        WARNING_LOG("FFT Handler already initialized, skipping...\n");
        return;
    }
    
    INFO_LOG("[FFT] Initializing FFT handler...\n");
    DEBUG_LOG("  Signal size (N): %zu\n", N);
    DEBUG_LOG("  Num shifts: %d\n", num_shifts);
    DEBUG_LOG("  Num signals: %d\n", num_signals);
    DEBUG_LOG("  Num output points (n_kg): %d\n", n_kg);
    DEBUG_LOG("  Scale factor: %.2e\n\n", scale_factor);
    
    // Сохранить параметры для использования в getFFTSize() и других методах
    fft_size_ = N;
    num_shifts_ = num_shifts;
    num_signals_ = num_signals;
    n_kg_ = n_kg;
    scale_factor_ = scale_factor;
    
    cl_int err = CL_SUCCESS;
    
    // ========================================================================
    // 1. CREATE GPU BUFFERS
    // ========================================================================
    
    DEBUG_LOG("[FFT] Allocating GPU buffers...\n");
    
    // Reference signals buffers
    ctx_.reference_data = clCreateBuffer(
        ctx_.context,
        CL_MEM_READ_WRITE,
        N * sizeof(int32_t),
        nullptr,
        &err
    );
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to allocate reference_data buffer");
    
    ctx_.reference_fft = clCreateBuffer(
        ctx_.context,
        CL_MEM_READ_WRITE,
        num_shifts * N * sizeof(cl_float2),
        nullptr,
        &err
    );
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to allocate reference_fft buffer");
    
    // Input signals buffers
    ctx_.input_data = clCreateBuffer(
        ctx_.context,
        CL_MEM_READ_WRITE,
        num_signals * N * sizeof(int32_t),
        nullptr,
        &err
    );
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to allocate input_data buffer");
    
    ctx_.input_fft = clCreateBuffer(
        ctx_.context,
        CL_MEM_READ_WRITE,
        num_signals * N * sizeof(cl_float2),
        nullptr,
        &err
    );
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to allocate input_fft buffer");
    
    // Correlation buffers
    ctx_.correlation_fft = clCreateBuffer(
        ctx_.context,
        CL_MEM_READ_WRITE,
        num_signals * num_shifts * N * sizeof(cl_float2),
        nullptr,
        &err
    );
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to allocate correlation_fft buffer");
    
    ctx_.correlation_ifft = clCreateBuffer(
        ctx_.context,
        CL_MEM_READ_WRITE,
        num_signals * num_shifts * N * sizeof(cl_float2),
        nullptr,
        &err
    );
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to allocate correlation_ifft buffer");
    
    DEBUG_LOG("[OK] GPU buffers allocated\n\n");
    
    // ========================================================================
    // 2. CREATE FFT PLANS (1D batch FFT)
    // ========================================================================
    
    DEBUG_LOG("[FFT] Creating FFT plans...\n");
    
    // Plan for reference signals (batch of num_shifts) with pre-callback (int32→float2) and post-callback (conjugate)
    ctx_.reference_fft_plan = create_fft_plan_1d_with_pre_and_post_callback_conjugate(N, num_shifts, scale_factor, "Reference FFT Plan");
    
    // Plan for input signals (batch of num_signals) with pre-callback
    ctx_.input_fft_plan = create_fft_plan_1d_with_precallback(N, num_signals, scale_factor, "Input FFT Plan");
    
    DEBUG_LOG("[OK] FFT plans created\n\n");
    
    // ========================================================================
    // 4. CREATE POST-CALLBACK USERDATA (must be created before IFFT plan)
    // ========================================================================
    
    DEBUG_LOG("[FFT] Creating post-callback userdata...\n");
    
    PostCallbackParams post_params = {
        (cl_uint)num_signals,
        (cl_uint)num_shifts,
        (cl_uint)N,
        (cl_uint)n_kg,
        (cl_uint)(N / 2)  // peak search range
    };
    
    create_post_callback_userdata(N, num_signals, num_shifts, n_kg, post_params);
    
    DEBUG_LOG("[OK] Post-callback userdata created\n\n");
    
    // Plan for correlation IFFT (batch of num_signals × num_shifts) 
    // with PRE-CALLBACK (Complex Multiply) and POST-CALLBACK (Find Peaks)
    // Оба callback'а встроены в план для минимального времени выполнения
    ctx_.correlation_ifft_plan = create_fft_plan_1d_with_pre_and_post_callback(N, num_signals * num_shifts, num_signals, num_shifts, n_kg, "Correlation IFFT Plan");
    
    DEBUG_LOG("[OK] IFFT plan with post-callback created\n\n");
    
    // ========================================================================
    // 3. CREATE PRE-CALLBACK USERDATA
    // ========================================================================
    
    DEBUG_LOG("[FFT] Creating pre-callback userdata...\n");
    
    PreCallbackParams pre_params = {
        (cl_uint)num_shifts,
        (cl_uint)N,
        0,  // is_hamming = 0 (no window for now)
        0   // scale_factor_exp
    };
    
    create_pre_callback_userdata(N, num_shifts, pre_params, nullptr);
    
    DEBUG_LOG("[OK] Pre-callback userdata created\n\n");
    
    ctx_.initialized = true;
    
    INFO_LOG("[OK] FFT Handler fully initialized!\n\n");
}

// ============================================================================
// Create 1D FFT Plan
// ============================================================================

clfftPlanHandle FFTHandler::create_fft_plan_1d(
    size_t fft_size,
    int batch_size,
    const std::string& plan_name
) {
    clfftPlanHandle plan_handle;
    cl_int err = CL_SUCCESS;
    
    size_t clLengths[1] = {fft_size};
    
    err = clfftCreateDefaultPlan(&plan_handle, ctx_.context, CLFFT_1D, clLengths);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("clfftCreateDefaultPlan failed for " + plan_name);
    }
    
    // Set FFT parameters
    clfftSetPlanPrecision(plan_handle, CLFFT_SINGLE);
    clfftSetLayout(plan_handle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    clfftSetResultLocation(plan_handle, CLFFT_OUTOFPLACE);
    clfftSetPlanBatchSize(plan_handle, batch_size);
    
    // Set strides and distances for batch processing
    size_t strides[1] = {1};
    size_t dist = fft_size;
    clfftSetPlanInStride(plan_handle, CLFFT_1D, strides);
    clfftSetPlanOutStride(plan_handle, CLFFT_1D, strides);
    clfftSetPlanDistance(plan_handle, dist, dist);
    
    // Bake the plan
    VERBOSE_LOG("  [DEBUG] Baking FFT plan: fft_size=%zu, batch_size=%d\n", fft_size, batch_size);
    err = clfftBakePlan(plan_handle, 1, &ctx_.queue, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        ERROR_LOG("clfftBakePlan failed for %s with error %d\n", plan_name.c_str(), err);
        throw std::runtime_error("clfftBakePlan failed for " + plan_name);
    }
    VERBOSE_LOG("  [DEBUG] FFT plan baked successfully\n");
    
    DEBUG_LOG("  ✓ %s created (size=%zu, batch=%d)\n", plan_name.c_str(), fft_size, batch_size);
    
    return plan_handle;
}

clfftPlanHandle FFTHandler::create_fft_plan_1d_with_precallback(
    size_t fft_size,
    int batch_size,
    float scale_factor,
    const std::string& plan_name
) {
    clfftPlanHandle plan_handle;
    cl_int err = CL_SUCCESS;
    
    size_t clLengths[1] = {fft_size};
    
    err = clfftCreateDefaultPlan(&plan_handle, ctx_.context, CLFFT_1D, clLengths);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("clfftCreateDefaultPlan failed for " + plan_name);
    }
    
    // Set FFT parameters
    clfftSetPlanPrecision(plan_handle, CLFFT_SINGLE);
    clfftSetLayout(plan_handle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    clfftSetResultLocation(plan_handle, CLFFT_OUTOFPLACE);
    clfftSetPlanBatchSize(plan_handle, batch_size);
    
    // Set strides and distances for batch processing
    size_t strides[1] = {1};
    size_t dist = fft_size;
    clfftSetPlanInStride(plan_handle, CLFFT_1D, strides);
    clfftSetPlanOutStride(plan_handle, CLFFT_1D, strides);
    clfftSetPlanDistance(plan_handle, dist, dist);
    
    // Load pre-callback function source (inline, matches clFFT callback signature)
    // Pre-callback signature: float2 callback_func(__global void* input, uint inoffset, __global void* userdata)
    // clFFT вызывает callback для каждого элемента отдельно, поэтому функция возвращает float2 для одного элемента
    std::string callback_func_source = R"(
typedef struct {
    float scale_factor;
    uint padding[3];  // Выравнивание до 16 байт
} PreCallbackParams;

float2 pre_callback(__global void* input, uint inoffset, __global void* userdata) {
    __global const int* in = (__global const int*)input;
    __global PreCallbackParams* params = (__global PreCallbackParams*)userdata;
    
    // Читаем int32 значение по индексу inoffset
    int val = in[inoffset];
    
    // Конвертируем в float2 с масштабированием
    float real = (float)val * params->scale_factor;
    float imag = 0.0f;  // Real signal - imaginary part is zero
    
    return (float2)(real, imag);
}
)";
    
    // Create userdata buffer with PreCallbackParams structure
    struct PreCallbackParams {
        float scale_factor;
        cl_uint padding[3];  // Выравнивание до 16 байт
    };
    PreCallbackParams pre_cb_params = {scale_factor, {0, 0, 0}};
    
    cl_mem callback_userdata = clCreateBuffer(ctx_.context, CL_MEM_READ_ONLY, sizeof(PreCallbackParams), nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create callback userdata buffer");
    }
    
    err = clEnqueueWriteBuffer(ctx_.queue, callback_userdata, CL_TRUE, 0, sizeof(PreCallbackParams), &pre_cb_params, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(callback_userdata);
        throw std::runtime_error("Failed to write callback userdata");
    }
    
    // Set pre-callback (must be done BEFORE BakePlan)
    // Note: callback function name should match the function name in the source string
    err = clfftSetPlanCallback(plan_handle, "pre_callback", callback_func_source.c_str(), 
                                0, PRECALLBACK, &callback_userdata, 1);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(callback_userdata);
        throw std::runtime_error("clfftSetPlanCallback failed for " + plan_name);
    }
    
    // Bake the plan (callback will be embedded)
    err = clfftBakePlan(plan_handle, 1, &ctx_.queue, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(callback_userdata);
        throw std::runtime_error("clfftBakePlan failed for " + plan_name);
    }
    
    // Note: callback_userdata is managed by clFFT plan, don't release it here
    
    DEBUG_LOG("  ✓ %s created with pre-callback (size=%zu, batch=%d)\n", plan_name.c_str(), fft_size, batch_size);
    
    return plan_handle;
}

clfftPlanHandle FFTHandler::create_fft_plan_1d_with_pre_and_post_callback_conjugate(
    size_t fft_size,
    int batch_size,
    float scale_factor,
    const std::string& plan_name
) {
    clfftPlanHandle plan_handle;
    cl_int err = CL_SUCCESS;
    
    size_t clLengths[1] = {fft_size};
    
    err = clfftCreateDefaultPlan(&plan_handle, ctx_.context, CLFFT_1D, clLengths);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("clfftCreateDefaultPlan failed for " + plan_name);
    }
    
    // Set FFT parameters
    clfftSetPlanPrecision(plan_handle, CLFFT_SINGLE);
    clfftSetLayout(plan_handle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    clfftSetResultLocation(plan_handle, CLFFT_OUTOFPLACE);
    clfftSetPlanBatchSize(plan_handle, batch_size);
    
    // Set strides and distances for batch processing
    size_t strides[1] = {1};
    size_t dist = fft_size;
    clfftSetPlanInStride(plan_handle, CLFFT_1D, strides);
    clfftSetPlanOutStride(plan_handle, CLFFT_1D, strides);
    clfftSetPlanDistance(plan_handle, dist, dist);
    
    // ========================================================================
    // PRE-CALLBACK: int32 → float2 conversion
    // ========================================================================
    std::string pre_callback_source = R"(
typedef struct {
    float scale_factor;
    uint padding[3];  // Выравнивание до 16 байт
} PreCallbackParams;

float2 pre_callback(__global void* input, uint inoffset, __global void* userdata) {
    __global const int* in = (__global const int*)input;
    __global PreCallbackParams* params = (__global PreCallbackParams*)userdata;
    
    // Читаем int32 значение по индексу inoffset
    int val = in[inoffset];
    
    // Конвертируем в float2 с масштабированием
    float real = (float)val * params->scale_factor;
    float imag = 0.0f;  // Real signal - imaginary part is zero
    
    return (float2)(real, imag);
}
)";
    
    // Create userdata buffer with PreCallbackParams structure
    struct PreCallbackParams {
        float scale_factor;
        cl_uint padding[3];  // Выравнивание до 16 байт
    };
    PreCallbackParams pre_cb_params = {scale_factor, {0, 0, 0}};
    
    cl_mem pre_callback_userdata = clCreateBuffer(ctx_.context, CL_MEM_READ_ONLY, sizeof(PreCallbackParams), nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create pre-callback userdata buffer");
    }
    
    err = clEnqueueWriteBuffer(ctx_.queue, pre_callback_userdata, CL_TRUE, 0, sizeof(PreCallbackParams), &pre_cb_params, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(pre_callback_userdata);
        throw std::runtime_error("Failed to write pre-callback userdata");
    }
    
    // Set pre-callback (must be done BEFORE BakePlan)
    err = clfftSetPlanCallback(plan_handle, "pre_callback", pre_callback_source.c_str(), 
                                0, PRECALLBACK, &pre_callback_userdata, 1);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(pre_callback_userdata);
        throw std::runtime_error("clfftSetPlanCallback failed for pre-callback in " + plan_name);
    }
    
    // ========================================================================
    // POST-CALLBACK: Complex Conjugate (real, imag) → (real, -imag)
    // ========================================================================
    std::string post_callback_source = R"(
void post_callback_conjugate(__global void* output, uint outoffset, __global void* userdata, float2 fftoutput) {
    __global float2* out = (__global float2*)output;
    // Комплексное сопряжение: (real, imag) → (real, -imag)
    out[outoffset] = (float2)(fftoutput.x, -fftoutput.y);
}
)";
    
    // Post-callback не требует userdata, так как модифицирует output напрямую
    // clFFT требует указатель на массив cl_mem, даже если он не используется
    // Создаем массив из одного nullptr элемента
    cl_mem post_callback_userdata_array[1] = {nullptr};
    
    // Set post-callback (must be done BEFORE BakePlan)
    // Последний параметр = 0, так как userdata не используется
    err = clfftSetPlanCallback(plan_handle, "post_callback_conjugate", post_callback_source.c_str(), 
                                0, POSTCALLBACK, post_callback_userdata_array, 0);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(pre_callback_userdata);
        throw std::runtime_error("clfftSetPlanCallback failed for post-callback in " + plan_name);
    }
    
    // Bake the plan (callbacks will be embedded)
    err = clfftBakePlan(plan_handle, 1, &ctx_.queue, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(pre_callback_userdata);
        throw std::runtime_error("clfftBakePlan failed for " + plan_name);
    }
    
    // Note: pre_callback_userdata is managed by clFFT plan, don't release it here
    
    DEBUG_LOG("  ✓ %s created with pre-callback (int32→float2) and post-callback (conjugate) (size=%zu, batch=%d)\n", 
           plan_name.c_str(), fft_size, batch_size);
    
    return plan_handle;
}

clfftPlanHandle FFTHandler::create_fft_plan_1d_with_postcallback(
    size_t fft_size,
    int batch_size,
    int num_signals,
    int num_shifts,
    int n_kg,
    const std::string& plan_name
) {
    clfftPlanHandle plan_handle;
    cl_int err = CL_SUCCESS;
    
    size_t clLengths[1] = {fft_size};
    
    err = clfftCreateDefaultPlan(&plan_handle, ctx_.context, CLFFT_1D, clLengths);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("clfftCreateDefaultPlan failed for " + plan_name);
    }
    
    // Set FFT parameters
    clfftSetPlanPrecision(plan_handle, CLFFT_SINGLE);
    clfftSetLayout(plan_handle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    clfftSetResultLocation(plan_handle, CLFFT_OUTOFPLACE);
    clfftSetPlanBatchSize(plan_handle, batch_size);
    
    // Set strides and distances for batch processing
    size_t strides[1] = {1};
    size_t dist = fft_size;
    clfftSetPlanInStride(plan_handle, CLFFT_1D, strides);
    clfftSetPlanOutStride(plan_handle, CLFFT_1D, strides);
    clfftSetPlanDistance(plan_handle, dist, dist);
    
    // Load post-callback function source (inline, matches clFFT callback signature)
    // Post-callback signature: void callback_func(__global void* output, uint outoffset, __global void* userdata, float2 fftoutput)
    // clFFT вызывает callback для каждого элемента отдельно после IFFT
    std::string callback_func_source = R"(
typedef struct {
    uint num_signals;
    uint num_shifts;
    uint fft_size;
    uint n_kg;
    uint search_range;
    uint padding[3];  // Выравнивание до 32 байт (8 uints)
} PostCallbackParams;

void post_callback(__global void* output, uint outoffset, __global void* userdata, float2 fftoutput) {
    __global PostCallbackParams* params = (__global PostCallbackParams*)userdata;
    __global float* peaks_output = (__global float*)((__global char*)userdata + sizeof(PostCallbackParams));
    
    uint num_signals = params->num_signals;
    uint num_shifts = params->num_shifts;
    uint fft_size = params->fft_size;
    uint n_kg = params->n_kg;
    uint search_range = params->search_range;
    
    // Вычислить индекс окна и позицию в окне
    uint window_idx = outoffset / fft_size;
    uint pos_in_window = outoffset % fft_size;
    
    if (window_idx >= num_signals * num_shifts) return;
    if (pos_in_window >= search_range) return;  // Ищем пик только в первой половине
    
    // Вычислить индексы сигнала и сдвига
    uint signal_idx = window_idx / num_shifts;
    uint shift_idx = window_idx % num_shifts;
    
    // Вычислить magnitude
    float magnitude = length(fftoutput);
    
    // Инициализируем максимум при первом элементе каждого окна
    if (pos_in_window == 0) {
        uint output_idx = (signal_idx * num_shifts + shift_idx) * n_kg;
        peaks_output[output_idx] = magnitude;
        // Остальные элементы заполняем нулями
        for (uint k = 1; k < n_kg; k++) {
            peaks_output[output_idx + k] = 0.0f;
        }
    } else {
        // Обновляем максимум если нашли больший
        uint output_idx = (signal_idx * num_shifts + shift_idx) * n_kg;
        if (magnitude > peaks_output[output_idx]) {
            peaks_output[output_idx] = magnitude;
        }
    }
}
)";
    
    // Use existing post_callback_userdata buffer
    cl_mem callback_userdata = ctx_.post_callback_userdata;
    if (!callback_userdata) {
        throw std::runtime_error("post_callback_userdata buffer not initialized");
    }
    
    // Set post-callback (must be done BEFORE BakePlan)
    err = clfftSetPlanCallback(plan_handle, "post_callback", callback_func_source.c_str(), 
                                0, POSTCALLBACK, &callback_userdata, 1);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("clfftSetPlanCallback failed for " + plan_name);
    }
    
    // Bake the plan (callback will be embedded)
    err = clfftBakePlan(plan_handle, 1, &ctx_.queue, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("clfftBakePlan failed for " + plan_name);
    }
    
    DEBUG_LOG("  ✓ %s created with post-callback (size=%zu, batch=%d)\n", plan_name.c_str(), fft_size, batch_size);
    
    return plan_handle;
}

// ============================================================================
// Create IFFT Plan with Pre-Callback (Complex Multiply) and Post-Callback (Find Peaks)
// ============================================================================

clfftPlanHandle FFTHandler::create_fft_plan_1d_with_pre_and_post_callback(
    size_t fft_size,
    int batch_size,
    int num_signals,
    int num_shifts,
    int n_kg,
    const std::string& plan_name
) {
    clfftPlanHandle plan_handle;
    cl_int err = CL_SUCCESS;
    
    size_t clLengths[1] = {fft_size};
    
    err = clfftCreateDefaultPlan(&plan_handle, ctx_.context, CLFFT_1D, clLengths);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("clfftCreateDefaultPlan failed for " + plan_name);
    }
    
    // Set FFT parameters
    clfftSetPlanPrecision(plan_handle, CLFFT_SINGLE);
    clfftSetLayout(plan_handle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    clfftSetResultLocation(plan_handle, CLFFT_OUTOFPLACE);
    clfftSetPlanBatchSize(plan_handle, batch_size);
    
    // Set strides and distances for batch processing
    size_t strides[1] = {1};
    size_t dist = fft_size;
    clfftSetPlanInStride(plan_handle, CLFFT_1D, strides);
    clfftSetPlanOutStride(plan_handle, CLFFT_1D, strides);
    clfftSetPlanDistance(plan_handle, dist, dist);
    
    // ========================================================================
    // PRE-CALLBACK: Complex Multiply
    // ========================================================================
    // PRE-CALLBACK signature: float2 pre_callback(__global void* input, uint inoffset, __global void* userdata)
    // Вызывается для каждого элемента входного буфера плана перед IFFT
    // Входной буфер плана - correlation_fft (будет заполнен результатом умножения)
    // userdata содержит параметры и указатели на reference_fft и input_fft
    std::string pre_callback_source = R"(
typedef struct {
    uint num_signals;
    uint num_shifts;
    uint fft_size;
    uint padding;
} ComplexMultiplyParams;

float2 pre_callback(__global void* input, uint inoffset, __global void* userdata) {
    __global ComplexMultiplyParams* params = (__global ComplexMultiplyParams*)userdata;
    __global float2* reference_fft = (__global float2*)((__global char*)userdata + sizeof(ComplexMultiplyParams));
    __global float2* input_fft = (__global float2*)((__global char*)userdata + sizeof(ComplexMultiplyParams) + sizeof(float2) * params->num_shifts * params->fft_size);
    
    uint num_signals = params->num_signals;
    uint num_shifts = params->num_shifts;
    uint fft_size = params->fft_size;
    
    // Вычислить индексы для correlation: total = num_signals * num_shifts * fft_size
    uint element_idx = inoffset % fft_size;
    uint window_idx = inoffset / fft_size;
    uint shift_idx = window_idx % num_shifts;
    uint signal_idx = window_idx / num_shifts;
    
    // Прочитать значения из reference_fft и input_fft (из userdata буфера)
    uint ref_idx = shift_idx * fft_size + element_idx;
    uint inp_idx = signal_idx * fft_size + element_idx;
    
    float2 ref = reference_fft[ref_idx];
    float2 inp = input_fft[inp_idx];
    
    // Complex multiply: result = ref * conj(inp)
    // (a + bi) * (c - di) = (ac + bd) + (bc - ad)i
    float real = ref.x * inp.x + ref.y * inp.y;
    float imag = ref.y * inp.x - ref.x * inp.y;
    
    return (float2)(real, imag);
}
)";
    
    // Create PRE-CALLBACK userdata buffer
    // Формат: параметры + reference_fft данные + input_fft данные (как в Native3Dcustom.cpp)
    struct ComplexMultiplyParams {
        cl_uint num_signals;
        cl_uint num_shifts;
        cl_uint fft_size;
        cl_uint padding;
    };
    
    ComplexMultiplyParams pre_params = {
        (cl_uint)num_signals,
        (cl_uint)num_shifts,
        (cl_uint)fft_size,
        0
    };
    
    size_t params_size = sizeof(ComplexMultiplyParams);
    size_t reference_size = num_shifts * fft_size * sizeof(cl_float2);
    size_t input_size = num_signals * fft_size * sizeof(cl_float2);
    size_t pre_cb_userdata_size = params_size + reference_size + input_size;
    
    cl_mem pre_callback_userdata = clCreateBuffer(
        ctx_.context,
        CL_MEM_READ_WRITE,
        pre_cb_userdata_size,
        nullptr,
        &err
    );
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create pre-callback userdata buffer");
    }
    
    // Записать параметры
    err = clEnqueueWriteBuffer(
        ctx_.queue,
        pre_callback_userdata,
        CL_TRUE,
        0,
        params_size,
        &pre_params,
        0, nullptr, nullptr
    );
    
    if (err != CL_SUCCESS) {
        clReleaseMemObject(pre_callback_userdata);
        throw std::runtime_error("Failed to write pre-callback params");
    }
    
    // Note: reference_fft и input_fft данные будут скопированы в userdata перед каждым вызовом IFFT
    // в функции step3_correlation (через clEnqueueCopyBuffer)
    
    // Set PRE-CALLBACK (must be done BEFORE BakePlan)
    err = clfftSetPlanCallback(plan_handle, "pre_callback", pre_callback_source.c_str(), 
                                0, PRECALLBACK, &pre_callback_userdata, 1);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(pre_callback_userdata);
        throw std::runtime_error("clfftSetPlanCallback failed for pre-callback in " + plan_name);
    }
    
    // Сохранить userdata в контексте для использования в step3
    ctx_.pre_callback_userdata_correlation = pre_callback_userdata;
    
    // ========================================================================
    // POST-CALLBACK: Find Peaks
    // ========================================================================
    std::string post_callback_source = R"(
typedef struct {
    uint num_signals;
    uint num_shifts;
    uint fft_size;
    uint n_kg;
    uint search_range;
    uint padding[1];
} PostCallbackParams;

void post_callback(__global void* output, uint outoffset, __global void* userdata, float2 fftoutput) {
    __global PostCallbackParams* params = (__global PostCallbackParams*)userdata;
    __global float* peaks_output = (__global float*)((__global char*)userdata + sizeof(PostCallbackParams));
    
    uint num_signals = params->num_signals;
    uint num_shifts = params->num_shifts;
    uint fft_size = params->fft_size;
    uint n_kg = params->n_kg;
    
    uint window_idx = outoffset / fft_size;
    uint pos_in_window = outoffset % fft_size;
    
    if (window_idx >= num_signals * num_shifts) return;
    if (pos_in_window >= n_kg) return;  // Выводим только первые n_kg значений
    
    uint signal_idx = window_idx / num_shifts;
    uint shift_idx = window_idx % num_shifts;
    
    // Вычислить магнитуду комплексного числа
    float magnitude = length(fftoutput);
    
    // Записать магнитуду в соответствующую позицию
    // Формат: M[signal_idx][shift_idx][pos_in_window]
    uint output_idx = (signal_idx * num_shifts + shift_idx) * n_kg;
    peaks_output[output_idx + pos_in_window] = magnitude;
}
)";
    
    // Use existing post_callback_userdata buffer
    cl_mem post_callback_userdata = ctx_.post_callback_userdata;
    if (!post_callback_userdata) {
        clReleaseMemObject(pre_callback_userdata);
        throw std::runtime_error("post_callback_userdata buffer not initialized");
    }
    
    // Set POST-CALLBACK
    err = clfftSetPlanCallback(plan_handle, "post_callback", post_callback_source.c_str(), 
                                0, POSTCALLBACK, &post_callback_userdata, 1);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(pre_callback_userdata);
        throw std::runtime_error("clfftSetPlanCallback failed for post-callback in " + plan_name);
    }
    
    // Bake the plan
    err = clfftBakePlan(plan_handle, 1, &ctx_.queue, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(pre_callback_userdata);
        throw std::runtime_error("clfftBakePlan failed for " + plan_name);
    }
    
    // Note: pre_callback_userdata is managed by clFFT plan, but we may need to release it
    // For now, keep it for potential future use
    
    DEBUG_LOG("  ✓ %s created with PRE-CALLBACK (Complex Multiply) and POST-CALLBACK (Find Peaks)\n", plan_name.c_str());
    DEBUG_LOG("    Note: Оба callback'а встроены в план для минимального времени выполнения!\n");
    
    return plan_handle;
}

// ============================================================================
// Create Pre-Callback Userdata
// ============================================================================

void FFTHandler::create_pre_callback_userdata(
    size_t N,
    int num_shifts,
    const PreCallbackParams& params,
    const float* hamming_window
) {
    // For now, just create a simple buffer with parameters
    // In real implementation, would include input signal data
    
    std::vector<cl_uint> params_vec = params.to_vector();
    
    size_t userdata_size = params_vec.size() * sizeof(cl_uint) 
                          + N * sizeof(int32_t);  // Space for input signal
    
    cl_int err = CL_SUCCESS;
    ctx_.pre_callback_userdata = clCreateBuffer(
        ctx_.context,
        CL_MEM_READ_WRITE,
        userdata_size,
        nullptr,
        &err
    );
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create pre_callback_userdata buffer");
    }
    
    // Write parameters
    err = clEnqueueWriteBuffer(
        ctx_.queue,
        ctx_.pre_callback_userdata,
        CL_TRUE,
        0,
        params_vec.size() * sizeof(cl_uint),
        params_vec.data(),
        0, nullptr, nullptr
    );
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to write pre_callback_userdata");
    }
}

// ============================================================================
// Create Post-Callback Userdata
// ============================================================================

void FFTHandler::create_post_callback_userdata(
    size_t N,
    int num_signals,
    int num_shifts,
    int n_kg,
    const PostCallbackParams& params
) {
    std::vector<cl_uint> params_vec = params.to_vector();
    
    // ВАЖНО: OpenCL структура PostCallbackParams имеет padding[1], поэтому размер структуры = 6 * sizeof(cl_uint) = 24 байта
    // params_vec содержит 5 элементов (n_signals, n_correlators, fft_size, n_kg, peak_search_range)
    // Но структура в OpenCL kernel имеет 6 элементов (5 параметров + padding[1])
    size_t params_size_in_buffer = 6 * sizeof(cl_uint);  // 5 параметров + padding[1] = 24 байта
    
    size_t output_size = num_signals * num_shifts * n_kg * sizeof(float);
    size_t userdata_size = params_size_in_buffer + output_size;
    
    cl_int err = CL_SUCCESS;
    ctx_.post_callback_userdata = clCreateBuffer(
        ctx_.context,
        CL_MEM_READ_WRITE,
        userdata_size,
        nullptr,
        &err
    );
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create post_callback_userdata buffer");
    }
    
    // Write parameters (без padding, padding заполнится нулями автоматически)
    // OpenCL структура ожидает padding[1], но мы записываем только параметры
    // Padding в буфере будет содержать мусор, но это OK, так как мы используем offset при чтении
    err = clEnqueueWriteBuffer(
        ctx_.queue,
        ctx_.post_callback_userdata,
        CL_TRUE,
        0,
        params_vec.size() * sizeof(cl_uint),
        params_vec.data(),
        0, nullptr, nullptr
    );
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to write post_callback_userdata");
    }
}

// ============================================================================
// STEP 1: Reference Signals + Forward FFT
// ============================================================================

void FFTHandler::step1_reference_signals(
    const int32_t* host_reference,
    size_t N,
    int num_shifts,
    float scale_factor,
    double& time_upload_ms,
    double& time_callback_ms,
    double& time_fft_ms,
    OperationTiming& upload_timing,
    OperationTiming& fft_timing
) {
    INFO_LOG("[STEP 1] Processing reference signals...\n");
    
    // ВАЖНО: Проверить соответствие параметров с параметрами инициализации
    VERBOSE_LOG("  [DEBUG] Step1 parameters check:\n");
    VERBOSE_LOG("    Passed: N=%zu, num_shifts=%d, scale_factor=%.6f\n", N, num_shifts, scale_factor);
    VERBOSE_LOG("    Stored: fft_size_=%zu, num_shifts_=%d, scale_factor_=%.6f\n", 
           fft_size_, num_shifts_, scale_factor_);
    
    if (N != fft_size_) {
        ERROR_LOG("N mismatch! Passed: %zu, Stored: %zu\n", N, fft_size_);
        throw std::runtime_error("FFT size mismatch in step1_reference_signals");
    }
    
    if (num_shifts != num_shifts_) {
        ERROR_LOG("num_shifts mismatch! Passed: %d, Stored: %d\n", num_shifts, num_shifts_);
        throw std::runtime_error("num_shifts mismatch in step1_reference_signals");
    }
    
    if (std::abs(scale_factor - scale_factor_) > 1e-6f) {
        WARNING_LOG("scale_factor mismatch! Passed: %.6f, Stored: %.6f\n", 
                scale_factor, scale_factor_);
        // Это не критично, но стоит проверить
    }
    
    // Проверить размеры буферов
    size_t expected_reference_data_size = fft_size_ * sizeof(int32_t);
    size_t expected_reference_fft_size = num_shifts_ * fft_size_ * sizeof(cl_float2);
    
    size_t actual_reference_data_size = 0;
    size_t actual_reference_fft_size = 0;
    
    if (ctx_.reference_data) {
        clGetMemObjectInfo(ctx_.reference_data, CL_MEM_SIZE, sizeof(size_t), &actual_reference_data_size, nullptr);
    }
    if (ctx_.reference_fft) {
        clGetMemObjectInfo(ctx_.reference_fft, CL_MEM_SIZE, sizeof(size_t), &actual_reference_fft_size, nullptr);
    }
    
    VERBOSE_LOG("  [DEBUG] Buffer sizes check:\n");
    VERBOSE_LOG("    reference_data: expected=%zu, actual=%zu\n", expected_reference_data_size, actual_reference_data_size);
    VERBOSE_LOG("    reference_fft: expected=%zu, actual=%zu\n", expected_reference_fft_size, actual_reference_fft_size);
    
    if (expected_reference_data_size != actual_reference_data_size) {
        ERROR_LOG("reference_data buffer size mismatch!\n");
        throw std::runtime_error("reference_data buffer size mismatch");
    }
    
    if (expected_reference_fft_size != actual_reference_fft_size) {
        ERROR_LOG("reference_fft buffer size mismatch!\n");
        throw std::runtime_error("reference_fft buffer size mismatch");
    }

    cl_int err = CL_SUCCESS;
    cl_event event_upload, event_fft;

    // ========================================================================
    // 1. Upload reference signal to GPU
    // ========================================================================

    DEBUG_LOG("  1. Uploading reference signal to GPU...\n");

    err = clEnqueueWriteBuffer(
        ctx_.queue,
        ctx_.reference_data,
        CL_FALSE,  // Non-blocking
        0,
        N * sizeof(int32_t),
        host_reference,
        0, nullptr,
        &event_upload
    );

    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to upload reference signal");
    }

    // Wait for upload and measure detailed time
    EventTiming upload_event_timing = profile_event_detailed(event_upload);
    time_upload_ms = upload_event_timing.execute_ms;
    upload_timing.execute_ms = upload_event_timing.execute_ms;
    upload_timing.queue_wait_ms = upload_event_timing.queue_wait_ms;
    upload_timing.cpu_wait_ms = upload_event_timing.wait_ms;
    upload_timing.total_gpu_ms = upload_event_timing.total_ms;
    DEBUG_LOG("  [PROFILE] Upload reference: execute=%.3f ms, queue_wait=%.3f ms, wait=%.3f ms\n", 
           upload_event_timing.execute_ms, upload_event_timing.queue_wait_ms, upload_event_timing.wait_ms);

    // ========================================================================
    // 2. Pre-Callback встроен в clFFT план (выполняется автоматически)
    // ========================================================================

    DEBUG_LOG("  2. Pre-callback встроен в clFFT план (выполняется автоматически)...\n");
    
    // Callback встроен в FFT, его время включено в FFT время
    time_callback_ms = 0.0;

    // ========================================================================
    // 3. Execute Forward FFT (callback выполнится автоматически через clFFT)
    // ========================================================================

    DEBUG_LOG("  3. Executing forward FFT (batch of %d) with embedded pre-callback...\n", num_shifts);

    // Проверить, что план валиден
    if (!ctx_.reference_fft_plan) {
        ERROR_LOG("reference_fft_plan is null!\n");
        throw std::runtime_error("reference_fft_plan is null");
    }
    
    // Проверить, что буферы валидны
    if (!ctx_.reference_data) {
        ERROR_LOG("reference_data buffer is null!\n");
        throw std::runtime_error("reference_data buffer is null");
    }
    
    if (!ctx_.reference_fft) {
        ERROR_LOG("reference_fft buffer is null!\n");
        throw std::runtime_error("reference_fft buffer is null");
    }
    
    VERBOSE_LOG("  [DEBUG] Plan and buffers check: plan=%p, ref_data=%p, ref_fft=%p\n",
           (void*)ctx_.reference_fft_plan, (void*)ctx_.reference_data, (void*)ctx_.reference_fft);

    // Инициализировать event_fft как nullptr перед вызовом
    event_fft = nullptr;
    
    VERBOSE_LOG("  [DEBUG] Calling clfftEnqueueTransform: plan=%p, queue=%p, input=%p, output=%p\n",
           (void*)ctx_.reference_fft_plan, (void*)ctx_.queue, 
           (void*)ctx_.reference_data, (void*)ctx_.reference_fft);

    clfftStatus fft_status = clfftEnqueueTransform(
        ctx_.reference_fft_plan,
        CLFFT_FORWARD,
        1,
        &ctx_.queue,
        1,  // Wait for upload to complete
        &event_upload,
        &event_fft,
        &ctx_.reference_data,  // Input: int32 data (callback конвертирует в float2)
        &ctx_.reference_fft,   // Output: float2 FFT results
        nullptr
    );

    VERBOSE_LOG("  [DEBUG] clfftEnqueueTransform status: %d (CLFFT_SUCCESS=%d)\n", fft_status, CLFFT_SUCCESS);
    VERBOSE_LOG("  [DEBUG] event_fft after enqueue: %p\n", (void*)event_fft);

    if (fft_status != CLFFT_SUCCESS) {
        ERROR_LOG("clfftEnqueueTransform failed with status %d\n", fft_status);
        if (event_upload) clReleaseEvent(event_upload);
        if (event_fft) clReleaseEvent(event_fft);
        throw std::runtime_error("clfftEnqueueTransform failed for reference FFT");
    }

    // Wait for FFT and measure detailed time
    if (event_fft) {
        // Проверить статус события перед профилированием
        cl_int event_status = CL_QUEUED;
        cl_int err = clGetEventInfo(event_fft, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &event_status, nullptr);
        if (err == CL_SUCCESS) {
            VERBOSE_LOG("  [DEBUG] FFT event status: %d (CL_COMPLETE=%d)\n", event_status, CL_COMPLETE);
        }
        
        EventTiming fft_event_timing = profile_event_detailed(event_fft);
        time_fft_ms = fft_event_timing.execute_ms;
        fft_timing.execute_ms = fft_event_timing.execute_ms;
        fft_timing.queue_wait_ms = fft_event_timing.queue_wait_ms;
        fft_timing.cpu_wait_ms = fft_event_timing.wait_ms;
        fft_timing.total_gpu_ms = fft_event_timing.total_ms;
        DEBUG_LOG("  [PROFILE] Forward FFT: execute=%.3f ms, queue_wait=%.3f ms, wait=%.3f ms\n", 
               fft_event_timing.execute_ms, fft_event_timing.queue_wait_ms, fft_event_timing.wait_ms);
        
        // Проверить, что время не равно нулю (это может означать, что операция не выполнилась)
        if (fft_event_timing.execute_ms == 0.0 && fft_event_timing.total_ms == 0.0) {
            WARNING_LOG("FFT timing is zero! This may indicate the operation did not execute.\n");
            WARNING_LOG("Check if the FFT plan is valid and buffers are correct.\n");
        }
        
        // Важно: дождаться завершения FFT операции перед освобождением события
        // Это гарантирует, что данные в ctx_.reference_fft готовы для чтения
        cl_int wait_err = clWaitForEvents(1, &event_fft);
        if (wait_err != CL_SUCCESS) {
            WARNING_LOG("clWaitForEvents failed with code %d\n", wait_err);
    } else {
            VERBOSE_LOG("  [DEBUG] FFT event completed successfully\n");
        }
    } else {
        ERROR_LOG("FFT event is null! clfftEnqueueTransform did not create an event.\n");
        ERROR_LOG("This means the FFT operation may not have been queued.\n");
        time_fft_ms = 0.0;
        fft_timing = OperationTiming{};
    }

    // Release events
    clReleaseEvent(event_upload);
    if (event_fft) clReleaseEvent(event_fft);
    
    // Дополнительно: убедиться, что все операции в очереди завершены
    // Это важно для гарантии, что данные готовы для чтения
    cl_int finish_err = clFinish(ctx_.queue);
    if (finish_err != CL_SUCCESS) {
        WARNING_LOG("clFinish failed with code %d after Step 1\n", finish_err);
    }

    INFO_LOG("[OK] Step 1 completed!\n\n");
}

// ============================================================================
// STEP 2: Input Signals + Forward FFT
// ============================================================================

void FFTHandler::step2_input_signals(
    const int32_t* host_input,
    size_t N,
    int num_signals,
    float scale_factor,
    double& time_upload_ms,
    double& time_callback_ms,
    double& time_fft_ms,
    OperationTiming& upload_timing,
    OperationTiming& fft_timing
) {
    INFO_LOG("[STEP 2] Processing input signals...\n");

    cl_int err = CL_SUCCESS;
    cl_event event_upload, event_fft;

    // Upload input signals
    DEBUG_LOG("  1. Uploading input signals to GPU...\n");


    err = clEnqueueWriteBuffer(
        ctx_.queue,
        ctx_.input_data,
        CL_FALSE,
        0,
        num_signals * N * sizeof(int32_t),
        host_input,
        0, nullptr,
        &event_upload
    );

    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to upload input signals");
    }

    // Wait for upload to complete and measure detailed time
    EventTiming upload_event_timing = profile_event_detailed(event_upload);
    time_upload_ms = upload_event_timing.execute_ms;
    upload_timing.execute_ms = upload_event_timing.execute_ms;
    upload_timing.queue_wait_ms = upload_event_timing.queue_wait_ms;
    upload_timing.cpu_wait_ms = upload_event_timing.wait_ms;
    upload_timing.total_gpu_ms = upload_event_timing.total_ms;
    DEBUG_LOG("  [PROFILE] Upload input: execute=%.3f ms, queue_wait=%.3f ms, wait=%.3f ms\n", 
           upload_event_timing.execute_ms, upload_event_timing.queue_wait_ms, upload_event_timing.wait_ms);

    // Pre-callback встроен в clFFT план, выполняется автоматически
    // Измеряем только время выполнения FFT (callback включен в FFT время)
    DEBUG_LOG("  2. Pre-callback встроен в clFFT план (выполняется автоматически)...\n");
    
    // Время callback включается в общее время FFT, так как он встроен
    // Для отдельного измерения нужно было бы использовать события, но callback встроен
    time_callback_ms = 0.0;  // Callback встроен в FFT, измеряется как часть FFT

    // Execute Forward FFT (callback выполнится автоматически через clFFT)
    DEBUG_LOG("  3. Executing forward FFT (batch of %d) with embedded pre-callback...\n", num_signals);

    clfftStatus fft_status = clfftEnqueueTransform(
        ctx_.input_fft_plan,
        CLFFT_FORWARD,
        1,
        &ctx_.queue,
        1,  // Wait for upload to complete
        &event_upload,
        &event_fft,
        &ctx_.input_data,  // Input: int32 data (callback конвертирует в float2)
        &ctx_.input_fft,   // Output: float2 FFT results
        nullptr
    );

    VERBOSE_LOG("  FFT status: %d\n", fft_status);

    if (fft_status != CLFFT_SUCCESS) {
        clReleaseEvent(event_upload);
        char error_msg[256];
        snprintf(error_msg, sizeof(error_msg), "clfftEnqueueTransform failed for input FFT: %d", fft_status);
        throw std::runtime_error(error_msg);
    }

    if (!event_fft) {
        clReleaseEvent(event_upload);
        throw std::runtime_error("FFT event is null");
    }

    // Wait for FFT to complete and measure time
    if (event_fft) {
        err = clWaitForEvents(1, &event_fft);
        if (err != CL_SUCCESS) {
            clReleaseEvent(event_upload);
            clReleaseEvent(event_fft);
            throw std::runtime_error("Failed to wait for FFT completion");
        }
    } else {
        WARNING_LOG("FFT event is null, skipping wait\n");
    }

    // Measure FFT time (detailed)
    if (event_fft) {
        EventTiming fft_event_timing = profile_event_detailed(event_fft);
        time_fft_ms = fft_event_timing.execute_ms;
        fft_timing.execute_ms = fft_event_timing.execute_ms;
        fft_timing.queue_wait_ms = fft_event_timing.queue_wait_ms;
        fft_timing.cpu_wait_ms = fft_event_timing.wait_ms;
        fft_timing.total_gpu_ms = fft_event_timing.total_ms;
        DEBUG_LOG("  [PROFILE] Forward FFT: execute=%.3f ms, queue_wait=%.3f ms, wait=%.3f ms\n", 
               fft_event_timing.execute_ms, fft_event_timing.queue_wait_ms, fft_event_timing.wait_ms);
    } else {
        time_fft_ms = 0.0;
        fft_timing = OperationTiming{};
    }

    // Clean up events
    clReleaseEvent(event_upload);
    if (event_fft) clReleaseEvent(event_fft);

    INFO_LOG("[OK] Step 2 completed!\n\n");
}

// ============================================================================
// STEP 3: Correlation (Multiply + IFFT + Post-callback)
// ============================================================================

void FFTHandler::step3_correlation(
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
) {
    INFO_LOG("[STEP 3] Computing correlation...\n");
    DEBUG_LOG("  Total correlations: %d × %d = %d\n", num_signals, num_shifts, num_signals * num_shifts);
    DEBUG_LOG("  Operation: 1. Pre-callback (Complex Multiply) → 2. IFFT → 3. Post-callback (Find Peaks) → 4. Download results\n\n");
    
    cl_int err = CL_SUCCESS;
    cl_event event_copy_data = nullptr, event_ifft = nullptr, event_download = nullptr;
    
    // Initialize output times to 0
    time_multiply_ms = 0.0;  // Теперь это время копирования данных в userdata (включено в IFFT время)
    time_ifft_ms = 0.0;
    time_download_ms = 0.0;
    time_post_callback_ms = 0.0;
    
    // ========================================================================
    // 1. PRE-CALLBACK: Подготовка данных для Complex Multiply
    // ========================================================================
    // Данные УЖЕ на GPU (ctx_.reference_fft из Step 1, ctx_.input_fft из Step 2)
    // Копируем их в userdata (GPU->GPU копирование) для PRE-CALLBACK
    // clFFT PRE-CALLBACK может читать данные только из userdata буфера
    // PRE-CALLBACK автоматически выполнит Complex Multiply при вызове IFFT
    
    DEBUG_LOG("  1. Pre-callback: Preparing data from GPU buffers (reference_fft + input_fft) for Complex Multiply...\n");
    DEBUG_LOG("     Note: Данные уже на GPU, выполняем быстрое GPU->GPU копирование в userdata\n");
    
    if (!ctx_.pre_callback_userdata_correlation) {
        throw std::runtime_error("pre_callback_userdata_correlation not initialized");
    }
    
    if (!ctx_.reference_fft || !ctx_.input_fft) {
        throw std::runtime_error("reference_fft or input_fft buffers not initialized (call Step 1 and Step 2 first)");
    }
    
    // Вычислить размеры и смещения
    // Используем тот же тип структуры, что и в create_fft_plan_1d_with_pre_and_post_callback
    struct ComplexMultiplyParams {
        cl_uint num_signals;
        cl_uint num_shifts;
        cl_uint fft_size;
        cl_uint padding;
    };
    size_t params_size = sizeof(ComplexMultiplyParams);
    size_t reference_size = num_shifts * N * sizeof(cl_float2);
    size_t input_size = num_signals * N * sizeof(cl_float2);
    
    // Копировать reference_fft (уже на GPU из Step 1) в userdata (после параметров)
    cl_event event_copy_ref = nullptr;
    err = clEnqueueCopyBuffer(
        ctx_.queue,                              // command_queue
        ctx_.reference_fft,                      // src_buffer
        ctx_.pre_callback_userdata_correlation,  // dst_buffer
        0,                                       // src_offset
        params_size,                             // dst_offset (после параметров)
        reference_size,                          // size
        0,                                       // num_events_in_wait_list
        nullptr,                                 // event_wait_list
        &event_copy_ref                          // event
    );
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to copy reference_fft to userdata");
    }
    
    // Копировать input_fft (уже на GPU из Step 2) в userdata (после reference_fft)
    err = clEnqueueCopyBuffer(
        ctx_.queue,                              // command_queue
        ctx_.input_fft,                          // src_buffer (уже на GPU)
        ctx_.pre_callback_userdata_correlation,  // dst_buffer (userdata для PRE-CALLBACK)
        0,                                       // src_offset
        params_size + reference_size,            // dst_offset (после параметров + reference_fft)
        input_size,                              // size
        1,                                       // num_events_in_wait_list
        &event_copy_ref,                         // event_wait_list (ждать копирования reference_fft)
        &event_copy_data                         // event
    );
    if (err != CL_SUCCESS) {
        clReleaseEvent(event_copy_ref);
        throw std::runtime_error("Failed to copy input_fft to userdata");
    }
    
    // Measure copy time (detailed) - это GPU->GPU копирование, очень быстрое
    EventTiming copy_event_timing = profile_event_detailed(event_copy_data);
    time_multiply_ms = copy_event_timing.execute_ms;  // Время GPU->GPU копирования
    multiply_timing.execute_ms = copy_event_timing.execute_ms;
    multiply_timing.queue_wait_ms = copy_event_timing.queue_wait_ms;
    multiply_timing.cpu_wait_ms = copy_event_timing.wait_ms;
    multiply_timing.total_gpu_ms = copy_event_timing.total_ms;
    DEBUG_LOG("  [PROFILE] GPU->GPU copy to userdata: execute=%.3f ms, queue_wait=%.3f ms, wait=%.3f ms\n", 
           copy_event_timing.execute_ms, copy_event_timing.queue_wait_ms, copy_event_timing.wait_ms);
    
    clReleaseEvent(event_copy_ref);
    
    DEBUG_LOG("  [OK] Data prepared in userdata (PRE-CALLBACK will perform Complex Multiply during IFFT)\n");
    
    // ========================================================================
    // 2. EXECUTE INVERSE FFT (PRE-CALLBACK и POST-CALLBACK встроены в план)
    // ========================================================================
    
    DEBUG_LOG("  2. Executing IFFT (batch of 2000) with embedded PRE-CALLBACK (Complex Multiply) and POST-CALLBACK (Find Peaks)...\n");
    
    // PRE-CALLBACK автоматически выполнит Complex Multiply из userdata
    // POST-CALLBACK автоматически запишет пики в post_callback_userdata
    clfftStatus fft_status = clfftEnqueueTransform(
        ctx_.correlation_ifft_plan,
        CLFFT_BACKWARD,
        1,
        &ctx_.queue,
        1,
        &event_copy_data,  // Ждать копирования данных в userdata
        &event_ifft,
        &ctx_.correlation_fft,  // Входной буфер (PRE-CALLBACK заполнит его результатом умножения)
        &ctx_.correlation_ifft,  // Выходной буфер (результаты IFFT)
        nullptr
    );
    
    if (fft_status != CLFFT_SUCCESS) {
        if (event_copy_data) clReleaseEvent(event_copy_data);
        throw std::runtime_error("clfftEnqueueTransform failed for correlation IFFT");
    }
    
    EventTiming ifft_event_timing = profile_event_detailed(event_ifft);
    time_ifft_ms = ifft_event_timing.execute_ms;
    ifft_timing.execute_ms = ifft_event_timing.execute_ms;
    ifft_timing.queue_wait_ms = ifft_event_timing.queue_wait_ms;
    ifft_timing.cpu_wait_ms = ifft_event_timing.wait_ms;
    ifft_timing.total_gpu_ms = ifft_event_timing.total_ms;
        DEBUG_LOG("  [PROFILE] Inverse FFT: execute=%.3f ms, queue_wait=%.3f ms, wait=%.3f ms\n", 
               ifft_event_timing.execute_ms, ifft_event_timing.queue_wait_ms, ifft_event_timing.wait_ms);
    
    // ========================================================================
    // 4. DOWNLOAD RESULTS
    // ========================================================================
    
    DEBUG_LOG("  3. Downloading correlation results (peaks) from POST-CALLBACK userdata...\n");
    DEBUG_LOG("     Size: %d × %d × %d elements = %.2f KB\n",
           num_signals, num_shifts, n_kg,
           num_signals * num_shifts * n_kg * sizeof(float) / 1024.0f);
    
    // Download peaks from post_callback_userdata (POST-CALLBACK уже записал туда пики)
    std::vector<float> peaks_data(num_signals * num_shifts * n_kg);
    
    // Вычислить смещение для данных в post_callback_userdata (после параметров)
    // ВАЖНО: Размер должен совпадать с OpenCL kernel структурой, которая имеет padding[1]
    // OpenCL struct: {uint num_signals, uint num_shifts, uint fft_size, uint n_kg, uint search_range, uint padding[1]}
    // = 6 * sizeof(uint) = 24 байта
    // Но при создании буфера использовался PostCallbackParams::to_vector() который возвращает 5 элементов = 20 байт
    // Нужно учесть padding[1], поэтому размер структуры = 6 * sizeof(cl_uint) = 24 байта
    size_t post_params_size = 6 * sizeof(cl_uint);  // n_signals, n_correlators, fft_size, n_kg, peak_search_range, padding[1]
    size_t peaks_size = num_signals * num_shifts * n_kg * sizeof(float);
    
    err = clEnqueueReadBuffer(
        ctx_.queue,
        ctx_.post_callback_userdata,  // Читаем из post_callback_userdata
        CL_FALSE,
        post_params_size,  // Смещение (после параметров)
        peaks_size,
        peaks_data.data(),
        1,
        &event_ifft,  // Ждать завершения IFFT (POST-CALLBACK выполнится внутри)
        &event_download
    );
    
    if (err != CL_SUCCESS) {
        ERROR_LOG("clEnqueueReadBuffer failed with error code: %d (CL_INVALID_VALUE)\n", err);
        ERROR_LOG("Details: offset=%zu, size=%zu, post_params_size=%zu\n", 
                post_params_size, peaks_size, post_params_size);
        if (!ctx_.post_callback_userdata) {
            ERROR_LOG("ctx_.post_callback_userdata is NULL!\n");
        }
        if (event_copy_data) clReleaseEvent(event_copy_data);
        if (event_ifft) clReleaseEvent(event_ifft);
        throw std::runtime_error("Failed to download results from post_callback_userdata");
    }
    
    EventTiming download_event_timing = profile_event_detailed(event_download);
    time_download_ms = download_event_timing.execute_ms;
    download_timing.execute_ms = download_event_timing.execute_ms;
    download_timing.queue_wait_ms = download_event_timing.queue_wait_ms;
    download_timing.cpu_wait_ms = download_event_timing.wait_ms;
    download_timing.total_gpu_ms = download_event_timing.total_ms;
    DEBUG_LOG("  [PROFILE] Download results: execute=%.3f ms, queue_wait=%.3f ms, wait=%.3f ms\n", 
           download_event_timing.execute_ms, download_event_timing.queue_wait_ms, download_event_timing.wait_ms);
    
    // Wait for download to complete
    if (event_download) {
        err = clWaitForEvents(1, &event_download);
        if (err != CL_SUCCESS) {
            if (event_copy_data) clReleaseEvent(event_copy_data);
            if (event_ifft) clReleaseEvent(event_ifft);
            if (event_download) clReleaseEvent(event_download);
            throw std::runtime_error("Failed to wait for download completion");
        }
    }
    
    // Post-callback (find peaks) встроен в IFFT план, выполняется автоматически
    // Извлечение пиков происходит внутри IFFT операции через clFFT callback
    DEBUG_LOG("  4. Post-callback (find peaks) встроен в IFFT план (выполняется автоматически)...\n");
    
    // Callback встроен в IFFT, его время включено в IFFT время
    time_post_callback_ms = 0.0;
    
    // Release events
    if (event_copy_data) clReleaseEvent(event_copy_data);
    if (event_ifft) clReleaseEvent(event_ifft);
    if (event_download) clReleaseEvent(event_download);
    
    INFO_LOG("\n[OK] Step 3 completed!\n");
    DEBUG_LOG("  Output: %d × %d × %d correlations\n",
           num_signals, num_shifts, n_kg);
    DEBUG_LOG("  Ready for results analysis\n\n");
}

// ============================================================================
// Get Correlation Results
// ============================================================================

std::vector<std::vector<std::vector<float>>> FFTHandler::get_correlation_results(
    int num_signals,
    int num_shifts,
    int n_kg
) {
    std::vector<std::vector<std::vector<float>>> results(
        num_signals,
        std::vector<std::vector<float>>(
            num_shifts,
            std::vector<float>(n_kg, 0.0f)
        )
    );
    
    // Download from GPU (will implement in step 3)
    
    return results;
}

// ============================================================================
// Cleanup
// ============================================================================

void FFTHandler::cleanup() {
  if(!ctx_.initialized)
    return;
  // ✅ ЗАЩИТА 1: Если уже вычищено - не трогаем!
  if (ctx_.is_cleaned_up) {
    DEBUG_LOG("[FFT] Already cleaned up, skipping...\n");
    return;  // ← ВЫХОД ЗДЕСЬ!
  }
    
  // ✅ ЗАЩИТА 2: Если не инициализировано - не трогаем!
  if (!ctx_.initialized) {
    DEBUG_LOG("[FFT] Not initialized, skipping cleanup\n");
    return;  // ← ВЫХОД ЗДЕСЬ!
  }
    

    INFO_LOG("[FFT] Cleaning up GPU resources...\n");
    
    // ========================================================================
    // 1. DESTROY FFT PLANS FIRST (ВАЖНО!)
    // ========================================================================
        
        
    DEBUG_LOG("  1. Destroying FFT plans...\n");
    
    if (ctx_.reference_fft_plan) {
        clfftStatus status = clfftDestroyPlan(&ctx_.reference_fft_plan);
        if (status == CLFFT_SUCCESS) {
            DEBUG_LOG("     ✓ Reference FFT plan destroyed\n");
        } else {
            WARNING_LOG("Failed to destroy reference FFT plan (code: %d)\n", status);
        }
        ctx_.reference_fft_plan = 0;
    }
    
    if (ctx_.input_fft_plan) {
        clfftStatus status = clfftDestroyPlan(&ctx_.input_fft_plan);
        if (status == CLFFT_SUCCESS) {
            DEBUG_LOG("     ✓ Input FFT plan destroyed\n");
        } else {
            WARNING_LOG("Failed to destroy input FFT plan (code: %d)\n", status);
        }
        ctx_.input_fft_plan = 0;
    }
    
    if (ctx_.correlation_ifft_plan) {
        clfftStatus status = clfftDestroyPlan(&ctx_.correlation_ifft_plan);
        if (status == CLFFT_SUCCESS) {
            DEBUG_LOG("     ✓ Correlation IFFT plan destroyed\n");
        } else {
            WARNING_LOG("Failed to destroy correlation IFFT plan (code: %d)\n", status);
        }
        ctx_.correlation_ifft_plan = 0;
    }
    
    // ========================================================================
    // 1.5. TEARDOWN clFFT LIBRARY (После уничтожения всех планов!)
    // ========================================================================
    
    DEBUG_LOG("  1.5. Tearing down clFFT library...\n");
    clfftStatus teardown_status = clfftTeardown();
    if (teardown_status == CLFFT_SUCCESS) {
        DEBUG_LOG("     ✓ clFFT library torn down\n");
    } else {
        WARNING_LOG("Failed to teardown clFFT library (code: %d)\n", teardown_status);
    }
    
    // ========================================================================
    // 2. RELEASE GPU MEMORY BUFFERS (После разрушения планов!)
    // ========================================================================
    
    DEBUG_LOG("  2. Releasing GPU memory buffers...\n");
    
    if (ctx_.reference_data) {
        cl_int status = clReleaseMemObject(ctx_.reference_data);
        if (status == CL_SUCCESS) {
            DEBUG_LOG("     ✓ Reference data buffer released\n");
        } else {
            WARNING_LOG("Failed to release reference data (code: %d)\n", status);
        }
        ctx_.reference_data = nullptr;
    }
    
    if (ctx_.reference_fft) {
        cl_int status = clReleaseMemObject(ctx_.reference_fft);
        if (status == CL_SUCCESS) {
            DEBUG_LOG("     ✓ Reference FFT buffer released\n");
        } else {
            WARNING_LOG("Failed to release reference FFT (code: %d)\n", status);
        }
        ctx_.reference_fft = nullptr;
    }
    
    if (ctx_.input_data) {
        cl_int status = clReleaseMemObject(ctx_.input_data);
        if (status == CL_SUCCESS) {
            DEBUG_LOG("     ✓ Input data buffer released\n");
        } else {
            WARNING_LOG("Failed to release input data (code: %d)\n", status);
        }
        ctx_.input_data = nullptr;
    }
    
    if (ctx_.input_fft) {
        cl_int status = clReleaseMemObject(ctx_.input_fft);
        if (status == CL_SUCCESS) {
            DEBUG_LOG("     ✓ Input FFT buffer released\n");
        } else {
            WARNING_LOG("Failed to release input FFT (code: %d)\n", status);
        }
        ctx_.input_fft = nullptr;
    }
    
    if (ctx_.correlation_fft) {
        cl_int status = clReleaseMemObject(ctx_.correlation_fft);
        if (status == CL_SUCCESS) {
            DEBUG_LOG("     ✓ Correlation FFT buffer released\n");
        } else {
            WARNING_LOG("Failed to release correlation FFT (code: %d)\n", status);
        }
        ctx_.correlation_fft = nullptr;
    }
    
    if (ctx_.correlation_ifft) {
        cl_int status = clReleaseMemObject(ctx_.correlation_ifft);
        if (status == CL_SUCCESS) {
            DEBUG_LOG("     ✓ Correlation IFFT buffer released\n");
        } else {
            WARNING_LOG("Failed to release correlation IFFT (code: %d)\n", status);
        }
        ctx_.correlation_ifft = nullptr;
    }
    
    if (ctx_.pre_callback_userdata) {
        cl_int status = clReleaseMemObject(ctx_.pre_callback_userdata);
        if (status == CL_SUCCESS) {
            DEBUG_LOG("     ✓ Pre-callback userdata buffer released\n");
        } else {
            WARNING_LOG("Failed to release pre-callback userdata (code: %d)\n", status);
        }
        ctx_.pre_callback_userdata = nullptr;
    }
    
    if (ctx_.post_callback_userdata) {
        cl_int status = clReleaseMemObject(ctx_.post_callback_userdata);
        if (status == CL_SUCCESS) {
            DEBUG_LOG("     ✓ Post-callback userdata buffer released\n");
        } else {
            WARNING_LOG("Failed to release post-callback userdata (code: %d)\n", status);
        }
        ctx_.post_callback_userdata = nullptr;
    }
    
    // ========================================================================
    // 3. MARK AS CLEANED UP (ВАЖНО!)
    // ========================================================================
    
    ctx_.initialized = false;
    ctx_.is_cleaned_up = true;  // ← СТАВИМ ФЛАГ!
    
    INFO_LOG("[OK] GPU cleanup complete!\n\n");
}

// ============================================================================
// Get Data Methods (for validation)
// ============================================================================

bool FFTHandler::getReferenceFFTData(std::vector<cl_float2>& output, int num_shifts, size_t fft_size) const {
    if (!ctx_.initialized) {
        ERROR_LOG("getReferenceFFTData - FFT handler not initialized\n");
        return false;
    }
    
    if (!ctx_.reference_fft) {
        ERROR_LOG("getReferenceFFTData - reference_fft buffer is null\n");
        return false;
    }
    
    if (!ctx_.queue) {
        ERROR_LOG("getReferenceFFTData - command queue is null\n");
        return false;
    }
    
    // Проверить, что handler не был очищен
    if (ctx_.is_cleaned_up) {
        ERROR_LOG("getReferenceFFTData - FFT handler already cleaned up\n");
        return false;
    }
    
    // Использовать сохраненные значения из initialize(), а не переданные параметры
    // Это гарантирует, что размер буфера совпадает с реальным размером
    int actual_num_shifts = num_shifts_;
    size_t actual_fft_size = fft_size_;
    
    // Проверить, что переданные параметры совпадают с сохраненными (для отладки)
    if (num_shifts != actual_num_shifts || fft_size != actual_fft_size) {
        WARNING_LOG("getReferenceFFTData - parameter mismatch!\n");
        VERBOSE_LOG("  Passed: num_shifts=%d, fft_size=%zu\n", num_shifts, fft_size);
        VERBOSE_LOG("  Actual: num_shifts=%d, fft_size=%zu\n", actual_num_shifts, actual_fft_size);
        VERBOSE_LOG("  Using actual values from initialize()\n");
    }
    
    // Проверить валидность контекста перед использованием
    if (!ctx_.context) {
        ERROR_LOG("getReferenceFFTData - context is null\n");
        return false;
    }
    
    // Вычислить ожидаемый размер на основе параметров
    size_t expected_data_size = actual_num_shifts * actual_fft_size;
    size_t expected_buffer_size = expected_data_size * sizeof(cl_float2);
    
    // Проверить реальный размер буфера перед чтением (это также проверяет валидность буфера)
    size_t actual_buffer_size = 0;
    cl_int err = clGetMemObjectInfo(ctx_.reference_fft, CL_MEM_SIZE, sizeof(size_t), &actual_buffer_size, nullptr);
    if (err != CL_SUCCESS) {
        ERROR_LOG("clGetMemObjectInfo failed with code %d - buffer may be invalid\n", err);
        ERROR_LOG("This usually means the buffer was released or the context is invalid\n");
        return false;
    }
    
    // Диагностика размеров
    VERBOSE_LOG("[DEBUG] getReferenceFFTData buffer sizes:\n");
    VERBOSE_LOG("  Expected: num_shifts=%d, fft_size=%zu, data_size=%zu, buffer_size=%zu bytes\n",
            actual_num_shifts, actual_fft_size, expected_data_size, expected_buffer_size);
    VERBOSE_LOG("  Actual buffer size: %zu bytes\n", actual_buffer_size);
    VERBOSE_LOG("  Size match: %s\n", (expected_buffer_size == actual_buffer_size) ? "YES" : "NO");
    
    if (expected_buffer_size > actual_buffer_size) {
        ERROR_LOG("Expected buffer size (%zu) exceeds actual buffer size (%zu)\n", 
                expected_buffer_size, actual_buffer_size);
        return false;
    }
    
    // ВАЖНО: Использовать реальный размер буфера, а не вычисленный!
    // clFFT может использовать padding или изменять размер буфера
    size_t buffer_size = actual_buffer_size;
    size_t data_size = actual_buffer_size / sizeof(cl_float2);
    
    if (expected_buffer_size != actual_buffer_size) {
        WARNING_LOG("Buffer size mismatch! Using actual buffer size.\n");
        VERBOSE_LOG("  Expected: %zu bytes, Actual: %zu bytes\n", expected_buffer_size, actual_buffer_size);
        VERBOSE_LOG("  This may indicate clFFT padding or buffer size changes\n");
    }
    
    output.resize(data_size);
    
    // Проверить, что очередь принадлежит тому же контексту, что и буфер
    cl_context buffer_context = nullptr;
    err = clGetMemObjectInfo(ctx_.reference_fft, CL_MEM_CONTEXT, sizeof(cl_context), &buffer_context, nullptr);
    if (err != CL_SUCCESS) {
        ERROR_LOG("clGetMemObjectInfo(CL_MEM_CONTEXT) failed with code %d\n", err);
        return false;
    }
    
    if (buffer_context != ctx_.context) {
        ERROR_LOG("getReferenceFFTData - buffer context mismatch!\n");
        VERBOSE_LOG("  Buffer context: %p, Handler context: %p\n", buffer_context, ctx_.context);
        return false;
    }
    
    // Если очередь невалидна, попробуем использовать clEnqueueReadBuffer напрямую
    // clEnqueueReadBuffer с CL_TRUE сам подождет завершения операций
    // Но если очередь невалидна, это не сработает
    if (!ctx_.queue) {
        ERROR_LOG("getReferenceFFTData - command queue is null\n");
        return false;
    }
    
    // Проверить, что очередь принадлежит тому же контексту
    cl_context queue_context = nullptr;
    err = clGetCommandQueueInfo(ctx_.queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &queue_context, nullptr);
    if (err != CL_SUCCESS) {
        WARNING_LOG("clGetCommandQueueInfo failed with code %d - queue may be invalid\n", err);
        // Если не можем проверить контекст очереди, но очередь не nullptr, попробуем использовать
    } else if (queue_context != ctx_.context) {
        ERROR_LOG("getReferenceFFTData - queue context mismatch!\n");
        VERBOSE_LOG("  Queue context: %p, Handler context: %p\n", queue_context, ctx_.context);
        return false;
    }
    
    // Попробовать прочитать буфер напрямую (clEnqueueReadBuffer с CL_TRUE сам подождет)
    // НЕ используем clFinish перед этим, так как clEnqueueReadBuffer с CL_TRUE сам синхронизирует
    err = clEnqueueReadBuffer(
        ctx_.queue,
        ctx_.reference_fft,
        CL_TRUE,  // Blocking read - автоматически ждет завершения операций
        0,
        buffer_size,
        output.data(),
        0, nullptr, nullptr
    );
    
    if (err != CL_SUCCESS) {
        ERROR_LOG("clEnqueueReadBuffer failed with code %d\n", err);
        if (err == CL_INVALID_MEM_OBJECT) {
            ERROR_LOG("  CL_INVALID_MEM_OBJECT (-5): Buffer is invalid or was released\n");
            VERBOSE_LOG("  Possible causes:\n");
            VERBOSE_LOG("    1. Buffer was released in cleanup()\n");
            VERBOSE_LOG("    2. Context was released\n");
            VERBOSE_LOG("    3. clFFT uses internal buffers (unlikely with CLFFT_OUTOFPLACE)\n");
        } else if (err == CL_INVALID_COMMAND_QUEUE) {
            ERROR_LOG("  CL_INVALID_COMMAND_QUEUE (-36): Command queue is invalid\n");
            VERBOSE_LOG("  Possible causes:\n");
            VERBOSE_LOG("    1. Queue was released in cleanup()\n");
            VERBOSE_LOG("    2. Context was released\n");
            VERBOSE_LOG("    3. Queue belongs to different context\n");
        }
        VERBOSE_LOG("  Actual buffer size: %zu bytes, Requested: %zu bytes\n", actual_buffer_size, buffer_size);
        VERBOSE_LOG("  num_shifts: %d, fft_size: %zu\n", actual_num_shifts, actual_fft_size);
        VERBOSE_LOG("  Context valid: %s, Queue valid: %s, Buffer valid: %s\n", 
                ctx_.context ? "yes" : "no", 
                ctx_.queue ? "yes" : "no",
                ctx_.reference_fft ? "yes" : "no");
        return false;
    }
    
    return true;
}

bool FFTHandler::getInputFFTData(std::vector<cl_float2>& output, int num_signals, size_t fft_size) const {
    if (!ctx_.initialized || !ctx_.input_fft) {
        ERROR_LOG("getInputFFTData - not initialized or buffer is null\n");
        return false;
    }
    
    if (!ctx_.queue) {
        ERROR_LOG("getInputFFTData - command queue is null\n");
        return false;
    }
    
    // Проверить, что handler не был очищен
    if (ctx_.is_cleaned_up) {
        ERROR_LOG("getInputFFTData - FFT handler already cleaned up\n");
        return false;
    }
    
    // Использовать сохраненные значения из initialize(), а не переданные параметры
    int actual_num_signals = num_signals_;
    size_t actual_fft_size = fft_size_;
    
    // Проверить, что переданные параметры совпадают с сохраненными (для отладки)
    if (num_signals != actual_num_signals || fft_size != actual_fft_size) {
        WARNING_LOG("getInputFFTData - parameter mismatch!\n");
        VERBOSE_LOG("  Passed: num_signals=%d, fft_size=%zu\n", num_signals, fft_size);
        VERBOSE_LOG("  Actual: num_signals=%d, fft_size=%zu\n", actual_num_signals, actual_fft_size);
        VERBOSE_LOG("  Using actual values from initialize()\n");
    }
    
    // Проверить валидность контекста перед использованием
    if (!ctx_.context) {
        ERROR_LOG("getInputFFTData - context is null\n");
        return false;
    }
    
    // Вычислить ожидаемый размер на основе параметров
    size_t expected_data_size = actual_num_signals * actual_fft_size;
    size_t expected_buffer_size = expected_data_size * sizeof(cl_float2);
    
    // Проверить реальный размер буфера перед чтением
    size_t actual_buffer_size = 0;
    cl_int err = clGetMemObjectInfo(ctx_.input_fft, CL_MEM_SIZE, sizeof(size_t), &actual_buffer_size, nullptr);
    if (err != CL_SUCCESS) {
        ERROR_LOG("clGetMemObjectInfo failed with code %d - buffer may be invalid\n", err);
        ERROR_LOG("This usually means the buffer was released or the context is invalid\n");
        return false;
    }
    
    // Диагностика размеров
    VERBOSE_LOG("[DEBUG] getInputFFTData buffer sizes:\n");
    VERBOSE_LOG("  Expected: num_signals=%d, fft_size=%zu, data_size=%zu, buffer_size=%zu bytes\n",
            actual_num_signals, actual_fft_size, expected_data_size, expected_buffer_size);
    VERBOSE_LOG("  Actual buffer size: %zu bytes\n", actual_buffer_size);
    VERBOSE_LOG("  Size match: %s\n", (expected_buffer_size == actual_buffer_size) ? "YES" : "NO");
    
    if (expected_buffer_size > actual_buffer_size) {
        ERROR_LOG("Expected buffer size (%zu) exceeds actual buffer size (%zu)\n", 
                expected_buffer_size, actual_buffer_size);
        return false;
    }
    
    // ВАЖНО: Использовать реальный размер буфера, а не вычисленный!
    size_t buffer_size = actual_buffer_size;
    size_t data_size = actual_buffer_size / sizeof(cl_float2);
    
    if (expected_buffer_size != actual_buffer_size) {
        WARNING_LOG("Buffer size mismatch! Using actual buffer size.\n");
        VERBOSE_LOG("  Expected: %zu bytes, Actual: %zu bytes\n", expected_buffer_size, actual_buffer_size);
        VERBOSE_LOG("  This may indicate clFFT padding or buffer size changes\n");
    }
    
    output.resize(data_size);
    
    // Попробовать прочитать буфер напрямую (clEnqueueReadBuffer с CL_TRUE сам подождет)
    err = clEnqueueReadBuffer(
        ctx_.queue,
        ctx_.input_fft,
        CL_TRUE,  // Blocking read - автоматически ждет завершения операций
        0,
        buffer_size,  // Использовать реальный размер буфера
        output.data(),
        0, nullptr, nullptr
    );
    
    if (err != CL_SUCCESS) {
        ERROR_LOG("clEnqueueReadBuffer failed with code %d\n", err);
        if (err == CL_INVALID_MEM_OBJECT) {
            ERROR_LOG("  CL_INVALID_MEM_OBJECT (-5): Buffer is invalid or was released\n");
        } else if (err == CL_INVALID_COMMAND_QUEUE) {
            ERROR_LOG("  CL_INVALID_COMMAND_QUEUE (-36): Command queue is invalid\n");
        }
        VERBOSE_LOG("  Actual buffer size: %zu bytes, Requested: %zu bytes\n", actual_buffer_size, buffer_size);
        VERBOSE_LOG("  num_signals: %d, fft_size: %zu\n", actual_num_signals, actual_fft_size);
        return false;
    }
    
    return true;
}

bool FFTHandler::getCorrelationPeaksData(std::vector<float>& output, int num_signals, int num_shifts, int n_kg) const {
    if (!ctx_.initialized || !ctx_.post_callback_userdata) {
        return false;
    }
    
    // Вычислить смещение для данных в post_callback_userdata (после параметров)
    struct PostCallbackParams {
        cl_uint n_signals;
        cl_uint n_correlators;
        cl_uint fft_size;
        cl_uint n_kg;
        cl_uint peak_search_range;
        cl_uint padding[1];
    };
    size_t post_params_size = 6 * sizeof(cl_uint);  // 5 параметров + padding[1]
    size_t peaks_size = num_signals * num_shifts * n_kg * sizeof(float);
    
    output.resize(num_signals * num_shifts * n_kg);
    
    cl_int err = clEnqueueReadBuffer(
        ctx_.queue,
        ctx_.post_callback_userdata,
        CL_TRUE,  // Blocking read
        post_params_size,  // Смещение (после параметров)
        peaks_size,
        output.data(),
        0, nullptr, nullptr
    );
    
    return err == CL_SUCCESS;
}

