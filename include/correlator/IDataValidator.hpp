#ifndef ICORRELATOR_DATA_VALIDATOR_HPP
#define ICORRELATOR_DATA_VALIDATOR_HPP

#include "IDataSnapshot.hpp"
#include "IConfiguration.hpp"
#include <string>
#include <vector>
#include <memory>

namespace Correlator {

/**
 * @struct ValidationResult
 * @brief Результат валидации данных
 */
struct ValidationResult {
    bool is_valid;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    std::string summary;

    ValidationResult() : is_valid(true) {}
    
    void addError(const std::string& error) {
        errors.push_back(error);
        is_valid = false;
    }
    
    void addWarning(const std::string& warning) {
        warnings.push_back(warning);
    }
    
    std::string toJSON() const;
};

/**
 * @class IDataValidator
 * @brief Интерфейс для валидации данных на каждом этапе
 * 
 * Проверяет корректность промежуточных результатов,
 * сравнивает с ожидаемыми значениями, выявляет аномалии.
 */
class IDataValidator {
public:
    virtual ~IDataValidator() = default;

    // Валидация на каждом этапе
    virtual ValidationResult validateStep1(const IDataSnapshot& snapshot, 
                                         const IConfiguration& config) const = 0;
    
    virtual ValidationResult validateStep2(const IDataSnapshot& snapshot, 
                                           const IConfiguration& config) const = 0;
    
    virtual ValidationResult validateStep3(const IDataSnapshot& snapshot, 
                                          const IConfiguration& config) const = 0;

    // Валидация промежуточных данных
    virtual ValidationResult validateReferenceFFT(const std::vector<ComplexFloat>& data,
                                                int num_shifts, size_t fft_size) const = 0;
    
    virtual ValidationResult validateInputFFT(const std::vector<ComplexFloat>& data,
                                             int num_signals, size_t fft_size) const = 0;
    
    virtual ValidationResult validateCorrelationFFT(const std::vector<ComplexFloat>& data,
                                                    int num_signals, int num_shifts, 
                                                    size_t fft_size) const = 0;
    
    virtual ValidationResult validateCorrelationIFFT(const std::vector<ComplexFloat>& data,
                                                     int num_signals, int num_shifts, 
                                                     size_t fft_size) const = 0;
    
    virtual ValidationResult validatePeaks(const std::vector<float>& peaks,
                                          int num_signals, int num_shifts, 
                                          int num_points) const = 0;

    // Сравнение с эталонными данными
    virtual ValidationResult compareWithReference(const IDataSnapshot& current,
                                                  const IDataSnapshot& reference) const = 0;

    // Экспорт результатов валидации в JSON
    virtual std::string exportValidationReport(const ValidationResult& result) const = 0;

    // Фабричный метод
    static std::unique_ptr<IDataValidator> createDefault();
};

} // namespace Correlator

#endif // ICORRELATOR_DATA_VALIDATOR_HPP

