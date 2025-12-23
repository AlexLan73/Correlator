#ifndef ICORRELATOR_FFT_BACKEND_HPP
#define ICORRELATOR_FFT_BACKEND_HPP

#include <vector>
#include <memory>
#include <cstdint>
#include "IDataSnapshot.hpp"
#include <CL/opencl.h>

namespace Correlator {

/**
 * @struct OperationTiming
 * @brief Детальная информация о времени выполнения операции
 */
struct OperationTiming {
    double execute_ms = 0.0;      // Время выполнения на GPU
    double queue_wait_ms = 0.0;   // Ожидание в очереди
    double cpu_wait_ms = 0.0;     // Ожидание CPU
    double total_gpu_ms = 0.0;    // Общее время GPU (QUEUED to END)
};

/**
 * @class IFFTBackend
 * @brief Интерфейс для FFT бэкенда (Strategy Pattern)
 * 
 * Абстрагирует реализацию FFT от конкретной платформы.
 * Позволяет легко переключаться между OpenCL, CUDA, ROCm и т.д.
 */
class IFFTBackend {
public:
    virtual ~IFFTBackend() = default;

    // Инициализация и очистка
    virtual bool initialize() = 0;
    virtual void cleanup() = 0;
    virtual bool isInitialized() const = 0;

    // Создание FFT планов
    virtual bool createReferenceFFTPlan(size_t fft_size, int batch_size, float scale_factor) = 0;
    virtual bool createInputFFTPlan(size_t fft_size, int batch_size, float scale_factor) = 0;
    virtual bool createCorrelationIFFTPlan(size_t fft_size, int batch_size, 
                                          int num_signals, int num_shifts, int n_kg) = 0;

    // Step 1: Обработка опорных сигналов
    virtual bool step1_ProcessReferenceSignals(
        const std::vector<int32_t>& reference_signal,
        int num_shifts,
        OperationTiming& upload_timing,
        OperationTiming& fft_timing
    ) = 0;

    // Step 2: Обработка входных сигналов
    virtual bool step2_ProcessInputSignals(
        const std::vector<int32_t>& input_signals,
        int num_signals,
        OperationTiming& upload_timing,
        OperationTiming& fft_timing
    ) = 0;

    // Step 3: Корреляция
    virtual bool step3_ComputeCorrelation(
        int num_signals,
        int num_shifts,
        int n_kg,
        OperationTiming& copy_timing,
        OperationTiming& ifft_timing,
        OperationTiming& download_timing
    ) = 0;

    // Получение результатов
    virtual bool getReferenceFFT(std::vector<ComplexFloat>& output) const = 0;
    virtual bool getInputFFT(std::vector<ComplexFloat>& output) const = 0;
    virtual bool getCorrelationPeaks(std::vector<float>& output) const = 0;

    // Информация о платформе
    virtual std::string getPlatformName() const = 0;
    virtual std::string getDeviceName() const = 0;
    virtual std::string getDriverVersion() const = 0;
    virtual std::string getAPIVersion() const = 0;
    
    // Получение device_id для профилирования (возвращает nullptr для не-OpenCL бэкендов)
    virtual cl_device_id getDeviceId() const = 0;

    // Фабричный метод
    static std::unique_ptr<IFFTBackend> createOpenCLBackend();
};

} // namespace Correlator

#endif // ICORRELATOR_FFT_BACKEND_HPP

