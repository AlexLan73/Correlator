#ifndef ICORRELATOR_DATA_SNAPSHOT_HPP
#define ICORRELATOR_DATA_SNAPSHOT_HPP

#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <complex>

namespace Correlator {

/**
 * @struct ComplexFloat
 * @brief Представление комплексного числа float
 */
struct ComplexFloat {
    float real;
    float imag;

    ComplexFloat() : real(0.0f), imag(0.0f) {}
    ComplexFloat(float r, float i) : real(r), imag(i) {}
    
    float magnitude() const {
        return std::sqrt(real * real + imag * imag);
    }
    
    float phase() const {
        return std::atan2(imag, real);
    }
};

/**
 * @class IDataSnapshot
 * @brief Интерфейс для сохранения промежуточных данных на каждом этапе
 * 
 * Используется для валидации и верификации алгоритма.
 * Поддерживает экспорт в JSON для анализа.
 */
class IDataSnapshot {
public:
    virtual ~IDataSnapshot() = default;

    // Метки этапов
    enum class Step {
        STEP1_REFERENCE_FFT,      // Спектры опорных сигналов
        STEP2_INPUT_FFT,          // Спектры входных сигналов
        STEP3_CORRELATION_FFT,    // Промежуточные спектры корреляции
        STEP3_CORRELATION_IFFT,   // Результаты IFFT
        STEP3_PEAKS               // Финальные пики
    };

    // Методы для сохранения данных
    virtual void saveReferenceFFT(const std::vector<ComplexFloat>& data, 
                                  int num_shifts, size_t fft_size) = 0;
    
    virtual void saveInputFFT(const std::vector<ComplexFloat>& data, 
                             int num_signals, size_t fft_size) = 0;
    
    virtual void saveCorrelationFFT(const std::vector<ComplexFloat>& data,
                                    int num_signals, int num_shifts, size_t fft_size) = 0;
    
    virtual void saveCorrelationIFFT(const std::vector<ComplexFloat>& data,
                                     int num_signals, int num_shifts, size_t fft_size) = 0;
    
    virtual void savePeaks(const std::vector<float>& peaks,
                          int num_signals, int num_shifts, int num_points) = 0;

    // Методы для получения данных
    virtual const std::vector<ComplexFloat>& getReferenceFFT() const = 0;
    virtual const std::vector<ComplexFloat>& getInputFFT() const = 0;
    virtual const std::vector<ComplexFloat>& getCorrelationFFT() const = 0;
    virtual const std::vector<ComplexFloat>& getCorrelationIFFT() const = 0;
    virtual const std::vector<float>& getPeaks() const = 0;

    // Метаданные
    virtual Step getStep() const = 0;
    virtual std::string getTimestamp() const = 0;
    virtual size_t getDataSize() const = 0;

    // Экспорт в JSON
    virtual std::string toJSON() const = 0;
    virtual std::string toJSON(Step step) const = 0;  // Экспорт конкретного этапа

    // Статистика
    virtual std::string getStatistics() const = 0;
};

} // namespace Correlator

#endif // ICORRELATOR_DATA_SNAPSHOT_HPP

