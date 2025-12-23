// ============================================================================
// FFT HANDLER HEADER - С ЗАЩИТОЙ ОТ ДВОЙНОГО CLEANUP
// ============================================================================

#ifndef FFT_HANDLER_HPP
#define FFT_HANDLER_HPP

#include <CL/opencl.h>
#include <clFFT.h>
#include <string>
#include <vector>

struct PreCallbackParams {
    cl_uint num_shifts;
    cl_uint N;
    cl_uint reserved1;
    cl_uint reserved2;
    
    std::vector<cl_uint> to_vector() const {
        return {num_shifts, N, reserved1, reserved2};
    }
};

struct PostCallbackParams {
    cl_uint num_signals;
    cl_uint num_shifts;
    cl_uint N;
    cl_uint n_kg;
    cl_uint peak_search_range;
    
    std::vector<cl_uint> to_vector() const {
        return {num_signals, num_shifts, N, n_kg, peak_search_range};
    }
};

struct FFTContext {
    cl_context context;
    cl_command_queue queue;
    cl_device_id device;
    
    cl_mem reference_data;
    cl_mem reference_fft;
    cl_mem input_data;
    cl_mem input_fft;
    cl_mem correlation_fft;
    cl_mem correlation_ifft;
    cl_mem pre_callback_userdata;
    cl_mem post_callback_userdata;
    
    clfftPlanHandle reference_fft_plan;
    clfftPlanHandle input_fft_plan;
    clfftPlanHandle correlation_ifft_plan;
    
    bool initialized;
    bool is_cleaned_up;  // ← НОВЫЙ ФЛАГ ЗАЩИТЫ!
    
    FFTContext() 
        : context(nullptr), queue(nullptr), device(nullptr),
          reference_data(nullptr), reference_fft(nullptr),
          input_data(nullptr), input_fft(nullptr),
          correlation_fft(nullptr), correlation_ifft(nullptr),
          pre_callback_userdata(nullptr), post_callback_userdata(nullptr),
          reference_fft_plan(nullptr), input_fft_plan(nullptr),
          correlation_ifft_plan(nullptr),
          initialized(false), is_cleaned_up(false)  // ← ИНИЦИАЛИЗИРУЕМ В FALSE
    {}
};

class FFTHandler {
public:
    FFTHandler(cl_context ctx, cl_command_queue q, cl_device_id dev);
    ~FFTHandler();
    
    void initialize(size_t N, int num_shifts, int num_signals, int n_kg, float scale_factor);
    void cleanup();  // Теперь безопасна к двойному вызову!
    
    void step1_reference_signals(const int32_t* host_reference, size_t N, int num_shifts,
                                 float scale_factor, double& time_upload_ms, 
                                 double& time_callback_ms, double& time_fft_ms);
    
    void step2_input_signals(const int32_t* host_input, size_t N, int num_signals,
                            float scale_factor, double& time_upload_ms,
                            double& time_callback_ms, double& time_fft_ms);
    
    void step3_correlation(int num_signals, int num_shifts, size_t N, int n_kg,
                          double& time_multiply_ms, double& time_ifft_ms,
                          double& time_download_ms);
    
    std::vector<std::vector<std::vector<float>>> get_correlation_results(
        int num_signals, int num_shifts, int n_kg);
    
private:
    FFTContext ctx_;
    
    clfftPlanHandle create_fft_plan_1d(size_t fft_size, int batch_size, const std::string& plan_name);
    void create_pre_callback_userdata(size_t N, int num_shifts, const PreCallbackParams& params,
                                      const float* hamming_window);
    void create_post_callback_userdata(size_t N, int num_signals, int num_shifts, int n_kg,
                                       const PostCallbackParams& params);
    
    double profile_event(cl_event event, const std::string& label);
};

#endif // FFT_HANDLER_HPP
