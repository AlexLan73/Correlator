#ifndef CORRELATOR_OPENCL_FFT_BACKEND_HPP
#define CORRELATOR_OPENCL_FFT_BACKEND_HPP

#include "IFFTBackend.hpp"
#include "../../include/fft_handler.hpp"
#include <CL/opencl.h>
#include <memory>
#include <vector>
#include <stdexcept>

namespace Correlator {

/**
 * @class OpenCLFFTBackend
 * @brief Реализация IFFTBackend для OpenCL платформы
 * 
 * Адаптер над существующим FFTHandler, реализующий интерфейс IFFTBackend.
 * Позволяет использовать существующий код в новой архитектуре.
 */
class OpenCLFFTBackend : public IFFTBackend {
private:
    std::unique_ptr<FFTHandler> fft_handler_;
    cl_context context_;
    cl_command_queue queue_;
    cl_device_id device_;
    bool initialized_;
    
    // Конфигурация для инициализации
    size_t fft_size_;
    int num_shifts_;
    int num_signals_;
    int n_kg_;
    float scale_factor_;

    // Внутренние буферы для хранения результатов
    mutable std::vector<ComplexFloat> reference_fft_cache_;
    mutable std::vector<ComplexFloat> input_fft_cache_;
    mutable std::vector<float> peaks_cache_;

    // Конвертация cl_float2 в ComplexFloat
    ComplexFloat toComplexFloat(const cl_float2& val) const {
        return ComplexFloat(val.s[0], val.s[1]);
    }

    // Конвертация ComplexFloat в cl_float2
    cl_float2 toCLFloat2(const ComplexFloat& val) const {
        cl_float2 result;
        result.s[0] = val.real;
        result.s[1] = val.imag;
        return result;
    }

public:
    OpenCLFFTBackend() 
        : initialized_(false), context_(nullptr), queue_(nullptr), device_(nullptr),
          fft_size_(32768), num_shifts_(40), num_signals_(50), n_kg_(5), 
          scale_factor_(1.0f / 32768.0f) {}
    
    // Метод для установки конфигурации перед initialize()
    void setConfiguration(size_t fft_size, int num_shifts, int num_signals, 
                         int n_kg, float scale_factor) {
        if (initialized_) {
            throw std::runtime_error("Cannot change configuration after initialization");
        }
        fft_size_ = fft_size;
        num_shifts_ = num_shifts;
        num_signals_ = num_signals;
        n_kg_ = n_kg;
        scale_factor_ = scale_factor;
    }

    bool initialize() override {
        if (initialized_) {
            return true;
        }

        try {
            // Инициализировать OpenCL контекст
            cl_int err = CL_SUCCESS;
            
            // Получить платформу
            cl_platform_id platform = nullptr;
            err = clGetPlatformIDs(1, &platform, nullptr);
            if (err != CL_SUCCESS) {
                return false;
            }
            
            // Получить устройство
            err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_, nullptr);
            if (err != CL_SUCCESS) {
                return false;
            }
            
            // Создать контекст
            context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
            if (err != CL_SUCCESS || !context_) {
                return false;
            }
            
            // Создать очередь команд
            queue_ = clCreateCommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE, &err);
            if (err != CL_SUCCESS || !queue_) {
                clReleaseContext(context_);
                return false;
            }
            
            // Создать FFTHandler
            fft_handler_ = std::make_unique<FFTHandler>(context_, queue_, device_);
            
            // Инициализировать FFTHandler (создать буферы и планы)
            fft_handler_->initialize(fft_size_, num_shifts_, num_signals_, n_kg_, scale_factor_);
            
            initialized_ = true;
            return true;
        } catch (...) {
            cleanup();
            return false;
        }
    }

    void cleanup() override {
        if (fft_handler_) {
            fft_handler_->cleanup();
            fft_handler_.reset();
        }
        if (queue_) {
            clReleaseCommandQueue(queue_);
            queue_ = nullptr;
        }
        if (context_) {
            clReleaseContext(context_);
            context_ = nullptr;
        }
        device_ = nullptr;
        initialized_ = false;
    }

    bool isInitialized() const override {
        return initialized_ && fft_handler_ != nullptr;
    }

    bool createReferenceFFTPlan(size_t fft_size, int batch_size, float scale_factor) override {
        if (!isInitialized()) {
            return false;
        }
        // Планы создаются в FFTHandler::initialize()
        // Здесь можно добавить дополнительную настройку если нужно
        return true;
    }

    bool createInputFFTPlan(size_t fft_size, int batch_size, float scale_factor) override {
        if (!isInitialized()) {
            return false;
        }
        return true;
    }

    bool createCorrelationIFFTPlan(size_t fft_size, int batch_size, 
                                  int num_signals, int num_shifts, int n_kg) override {
        if (!isInitialized()) {
            return false;
        }
        return true;
    }

    bool step1_ProcessReferenceSignals(
        const std::vector<int32_t>& reference_signal,
        int num_shifts,
        OperationTiming& upload_timing,
        OperationTiming& fft_timing
    ) override {
        if (!isInitialized()) {
            return false;
        }

        try {
            // Вызвать существующий метод FFTHandler
            // Нужно адаптировать сигнатуру под существующий код
            double time_upload = 0.0;
            double time_fft = 0.0;
            
            FFTHandler::OperationTiming upload_op_timing, fft_op_timing;
            
            double time_callback = 0.0;
            fft_handler_->step1_reference_signals(
                reference_signal.data(),
                reference_signal.size(),
                num_shifts,
                scale_factor_,
                time_upload,
                time_callback,
                time_fft,
                upload_op_timing,
                fft_op_timing
            );

            // Конвертация в OperationTiming
            upload_timing.execute_ms = upload_op_timing.execute_ms;
            upload_timing.queue_wait_ms = upload_op_timing.queue_wait_ms;
            upload_timing.cpu_wait_ms = upload_op_timing.cpu_wait_ms;
            upload_timing.total_gpu_ms = upload_op_timing.total_gpu_ms;

            fft_timing.execute_ms = fft_op_timing.execute_ms;
            fft_timing.queue_wait_ms = fft_op_timing.queue_wait_ms;
            fft_timing.cpu_wait_ms = fft_op_timing.cpu_wait_ms;
            fft_timing.total_gpu_ms = fft_op_timing.total_gpu_ms;

            // Очистить кэш, чтобы при следующем getReferenceFFT данные были свежими
            reference_fft_cache_.clear();

            return true;
        } catch (...) {
            return false;
        }
    }

    bool step2_ProcessInputSignals(
        const std::vector<int32_t>& input_signals,
        int num_signals,
        OperationTiming& upload_timing,
        OperationTiming& fft_timing
    ) override {
        if (!isInitialized()) {
            return false;
        }

        try {
            double time_upload = 0.0;
            double time_fft = 0.0;
            
            FFTHandler::OperationTiming upload_op_timing, fft_op_timing;
            
            double time_callback = 0.0;
            fft_handler_->step2_input_signals(
                input_signals.data(),
                input_signals.size()/num_signals,
                num_signals,
                scale_factor_,
                time_upload,
                time_callback,
                time_fft,
                upload_op_timing,
                fft_op_timing
            );

            // Конвертация в OperationTiming
            upload_timing.execute_ms = upload_op_timing.execute_ms;
            upload_timing.queue_wait_ms = upload_op_timing.queue_wait_ms;
            upload_timing.cpu_wait_ms = upload_op_timing.cpu_wait_ms;
            upload_timing.total_gpu_ms = upload_op_timing.total_gpu_ms;

            fft_timing.execute_ms = fft_op_timing.execute_ms;
            fft_timing.queue_wait_ms = fft_op_timing.queue_wait_ms;
            fft_timing.cpu_wait_ms = fft_op_timing.cpu_wait_ms;
            fft_timing.total_gpu_ms = fft_op_timing.total_gpu_ms;

            // Очистить кэш
            input_fft_cache_.clear();

            return true;
        } catch (...) {
            return false;
        }
    }

    bool step3_ComputeCorrelation(
        int num_signals,
        int num_shifts,
        int n_kg,
        OperationTiming& copy_timing,
        OperationTiming& ifft_timing,
        OperationTiming& download_timing
    ) override {
        if (!isInitialized()) {
            return false;
        }

        try {
            double time_multiply = 0.0;
            double time_ifft = 0.0;
            double time_download = 0.0;
            
            FFTHandler::OperationTiming multiply_timing, ifft_op_timing, download_op_timing;
            
            double time_post_callback = 0.0;
            fft_handler_->step3_correlation(
                num_signals,
                num_shifts,
                fft_size_,
                n_kg,
                time_multiply,
                time_ifft,
                time_download,
                time_post_callback,
                multiply_timing,
                ifft_op_timing,
                download_op_timing
            );

            // Конвертация
            copy_timing = {
                multiply_timing.execute_ms,
                multiply_timing.queue_wait_ms,
                multiply_timing.cpu_wait_ms,
                multiply_timing.total_gpu_ms
            };

            ifft_timing = {
                ifft_op_timing.execute_ms,
                ifft_op_timing.queue_wait_ms,
                ifft_op_timing.cpu_wait_ms,
                ifft_op_timing.total_gpu_ms
            };

            download_timing = {
                download_op_timing.execute_ms,
                download_op_timing.queue_wait_ms,
                download_op_timing.cpu_wait_ms,
                download_op_timing.total_gpu_ms
            };

            // Очистить кэш
            peaks_cache_.clear();

            return true;
        } catch (...) {
            return false;
        }
    }

    bool getReferenceFFT(std::vector<ComplexFloat>& output) const override {
        if (!isInitialized()) {
            return false;
        }

        // Загрузить данные из FFTHandler
        std::vector<cl_float2> cl_data;
        if (!fft_handler_->getReferenceFFTData(cl_data, num_shifts_, fft_size_)) {
            return false;
        }

        // Конвертировать в ComplexFloat
        output.clear();
        output.reserve(cl_data.size());
        for (const auto& val : cl_data) {
            output.push_back(toComplexFloat(val));
        }

        return true;
    }

    bool getInputFFT(std::vector<ComplexFloat>& output) const override {
        if (!isInitialized()) {
            return false;
        }

        // Загрузить данные из FFTHandler
        std::vector<cl_float2> cl_data;
        if (!fft_handler_->getInputFFTData(cl_data, num_signals_, fft_size_)) {
            return false;
        }

        // Конвертировать в ComplexFloat
        output.clear();
        output.reserve(cl_data.size());
        for (const auto& val : cl_data) {
            output.push_back(toComplexFloat(val));
        }

        return true;
    }

    bool getCorrelationPeaks(std::vector<float>& output) const override {
        if (!isInitialized()) {
            return false;
        }

        // Загрузить данные из FFTHandler
        return fft_handler_->getCorrelationPeaksData(output, num_signals_, num_shifts_, n_kg_);
    }

    std::string getPlatformName() const override {
        return "OpenCL";
    }

    std::string getDeviceName() const override {
        if (!device_) {
            return "Unknown";
        }
        
        char device_name[256] = {0};
        cl_int err = clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
        if (err != CL_SUCCESS) {
            return "Unknown";
        }
        return std::string(device_name);
    }

    std::string getDriverVersion() const override {
        if (!device_) {
            return "Unknown";
        }
        
        char driver_version[256] = {0};
        cl_int err = clGetDeviceInfo(device_, CL_DRIVER_VERSION, sizeof(driver_version), driver_version, nullptr);
        if (err != CL_SUCCESS) {
            return "Unknown";
        }
        return std::string(driver_version);
    }

    std::string getAPIVersion() const override {
        if (!device_) {
            return "Unknown";
        }
        
        char version[256] = {0};
        cl_int err = clGetDeviceInfo(device_, CL_DEVICE_VERSION, sizeof(version), version, nullptr);
        if (err != CL_SUCCESS) {
            return "Unknown";
        }
        return std::string(version);
    }

    cl_device_id getDeviceId() const override {
        return device_;
    }
};

// Фабричный метод
inline std::unique_ptr<IFFTBackend> IFFTBackend::createOpenCLBackend() {
    return std::make_unique<OpenCLFFTBackend>();
}

} // namespace Correlator

#endif // CORRELATOR_OPENCL_FFT_BACKEND_HPP

