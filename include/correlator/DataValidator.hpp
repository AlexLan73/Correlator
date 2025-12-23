#ifndef CORRELATOR_DATA_VALIDATOR_HPP
#define CORRELATOR_DATA_VALIDATOR_HPP

#include "IDataValidator.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace Correlator {

/**
 * @class DataValidator
 * @brief Реализация валидации данных
 * 
 * Проверяет корректность данных на каждом этапе обработки.
 * Выявляет аномалии, проверяет размеры, диапазоны значений.
 */
class DataValidator : public IDataValidator {
private:
    // Пороги для валидации
    static constexpr float MAX_MAGNITUDE = 1e6f;
    static constexpr float MIN_MAGNITUDE = 1e-9f;
    static constexpr float MAX_PEAK_VALUE = 1e6f;
    static constexpr float MIN_PEAK_VALUE = 0.0f;

public:
    ValidationResult validateStep1(const IDataSnapshot& snapshot, 
                                   const IConfiguration& config) const override {
        ValidationResult result;
        
        const auto& data = snapshot.getReferenceFFT();
        int expected_size = config.getNumShifts() * config.getFFTSize();
        
        if (data.size() != expected_size) {
            result.addError("Step 1: Reference FFT size mismatch. Expected: " + 
                          std::to_string(expected_size) + ", Got: " + 
                          std::to_string(data.size()));
        }
        
        auto fft_result = validateReferenceFFT(data, config.getNumShifts(), config.getFFTSize());
        if (!fft_result.is_valid) {
            result.errors.insert(result.errors.end(), 
                               fft_result.errors.begin(), fft_result.errors.end());
        }
        
        result.warnings.insert(result.warnings.end(),
                              fft_result.warnings.begin(), fft_result.warnings.end());
        
        return result;
    }

    ValidationResult validateStep2(const IDataSnapshot& snapshot, 
                                  const IConfiguration& config) const override {
        ValidationResult result;
        
        const auto& data = snapshot.getInputFFT();
        int expected_size = config.getNumSignals() * config.getFFTSize();
        
        if (data.size() != expected_size) {
            result.addError("Step 2: Input FFT size mismatch. Expected: " + 
                          std::to_string(expected_size) + ", Got: " + 
                          std::to_string(data.size()));
        }
        
        auto fft_result = validateInputFFT(data, config.getNumSignals(), config.getFFTSize());
        if (!fft_result.is_valid) {
            result.errors.insert(result.errors.end(), 
                               fft_result.errors.begin(), fft_result.errors.end());
        }
        
        result.warnings.insert(result.warnings.end(),
                              fft_result.warnings.begin(), fft_result.warnings.end());
        
        return result;
    }

    ValidationResult validateStep3(const IDataSnapshot& snapshot, 
                                  const IConfiguration& config) const override {
        ValidationResult result;
        
        // Валидация IFFT результатов
        const auto& ifft_data = snapshot.getCorrelationIFFT();
        int expected_ifft_size = config.getNumSignals() * config.getNumShifts() * config.getFFTSize();
        
        if (ifft_data.size() != expected_ifft_size) {
            result.addError("Step 3: Correlation IFFT size mismatch. Expected: " + 
                          std::to_string(expected_ifft_size) + ", Got: " + 
                          std::to_string(ifft_data.size()));
        }
        
        // Валидация пиков
        const auto& peaks = snapshot.getPeaks();
        int expected_peaks_size = config.getNumSignals() * config.getNumShifts() * config.getNumOutputPoints();
        
        if (peaks.size() != expected_peaks_size) {
            result.addError("Step 3: Peaks size mismatch. Expected: " + 
                          std::to_string(expected_peaks_size) + ", Got: " + 
                          std::to_string(peaks.size()));
        }
        
        auto peaks_result = validatePeaks(peaks, config.getNumSignals(), 
                                         config.getNumShifts(), config.getNumOutputPoints());
        if (!peaks_result.is_valid) {
            result.errors.insert(result.errors.end(), 
                               peaks_result.errors.begin(), peaks_result.errors.end());
        }
        
        return result;
    }

    ValidationResult validateReferenceFFT(const std::vector<ComplexFloat>& data,
                                          int num_shifts, size_t fft_size) const override {
        ValidationResult result;
        
        if (data.empty()) {
            result.addError("Reference FFT data is empty");
            return result;
        }
        
        int expected_size = num_shifts * fft_size;
        if (data.size() != expected_size) {
            result.addError("Reference FFT size mismatch");
            return result;
        }
        
        // Проверка диапазонов значений
        for (size_t i = 0; i < data.size(); ++i) {
            float mag = data[i].magnitude();
            if (std::isnan(mag) || std::isinf(mag)) {
                result.addError("Reference FFT contains NaN/Inf at index " + std::to_string(i));
            }
            if (mag > MAX_MAGNITUDE) {
                result.addWarning("Reference FFT magnitude too large at index " + std::to_string(i));
            }
        }
        
        return result;
    }

    ValidationResult validateInputFFT(const std::vector<ComplexFloat>& data,
                                      int num_signals, size_t fft_size) const override {
        ValidationResult result;
        
        if (data.empty()) {
            result.addError("Input FFT data is empty");
            return result;
        }
        
        int expected_size = num_signals * fft_size;
        if (data.size() != expected_size) {
            result.addError("Input FFT size mismatch");
            return result;
        }
        
        // Проверка диапазонов значений
        for (size_t i = 0; i < data.size(); ++i) {
            float mag = data[i].magnitude();
            if (std::isnan(mag) || std::isinf(mag)) {
                result.addError("Input FFT contains NaN/Inf at index " + std::to_string(i));
            }
            if (mag > MAX_MAGNITUDE) {
                result.addWarning("Input FFT magnitude too large at index " + std::to_string(i));
            }
        }
        
        return result;
    }

    ValidationResult validateCorrelationFFT(const std::vector<ComplexFloat>& data,
                                           int num_signals, int num_shifts, 
                                           size_t fft_size) const override {
        ValidationResult result;
        
        if (data.empty()) {
            result.addError("Correlation FFT data is empty");
            return result;
        }
        
        int expected_size = num_signals * num_shifts * fft_size;
        if (data.size() != expected_size) {
            result.addError("Correlation FFT size mismatch");
            return result;
        }
        
        return result;
    }

    ValidationResult validateCorrelationIFFT(const std::vector<ComplexFloat>& data,
                                             int num_signals, int num_shifts, 
                                             size_t fft_size) const override {
        ValidationResult result;
        
        if (data.empty()) {
            result.addError("Correlation IFFT data is empty");
            return result;
        }
        
        int expected_size = num_signals * num_shifts * fft_size;
        if (data.size() != expected_size) {
            result.addError("Correlation IFFT size mismatch");
            return result;
        }
        
        return result;
    }

    ValidationResult validatePeaks(const std::vector<float>& peaks,
                                  int num_signals, int num_shifts, 
                                  int num_points) const override {
        ValidationResult result;
        
        if (peaks.empty()) {
            result.addError("Peaks data is empty");
            return result;
        }
        
        int expected_size = num_signals * num_shifts * num_points;
        if (peaks.size() != expected_size) {
            result.addError("Peaks size mismatch");
            return result;
        }
        
        // Проверка диапазонов значений
        for (size_t i = 0; i < peaks.size(); ++i) {
            if (std::isnan(peaks[i]) || std::isinf(peaks[i])) {
                result.addError("Peaks contains NaN/Inf at index " + std::to_string(i));
            }
            if (peaks[i] < MIN_PEAK_VALUE || peaks[i] > MAX_PEAK_VALUE) {
                result.addWarning("Peak value out of expected range at index " + std::to_string(i));
            }
        }
        
        return result;
    }

    ValidationResult compareWithReference(const IDataSnapshot& current,
                                         const IDataSnapshot& reference) const override {
        ValidationResult result;
        // TODO: Реализовать сравнение с эталонными данными
        return result;
    }

    std::string exportValidationReport(const ValidationResult& result) const override {
        std::ostringstream oss;
        oss << "{\n"
            << "  \"is_valid\": " << (result.is_valid ? "true" : "false") << ",\n"
            << "  \"errors\": [";
        
        for (size_t i = 0; i < result.errors.size(); ++i) {
            if (i > 0) oss << ",";
            oss << "\n    \"" << result.errors[i] << "\"";
        }
        oss << "\n  ],\n"
            << "  \"warnings\": [";
        
        for (size_t i = 0; i < result.warnings.size(); ++i) {
            if (i > 0) oss << ",";
            oss << "\n    \"" << result.warnings[i] << "\"";
        }
        oss << "\n  ]\n"
            << "}";
        
        return oss.str();
    }
};

// Реализация ValidationResult::toJSON
inline std::string ValidationResult::toJSON() const {
    std::ostringstream oss;
    oss << "{\n"
        << "  \"is_valid\": " << (is_valid ? "true" : "false") << ",\n"
        << "  \"error_count\": " << errors.size() << ",\n"
        << "  \"warning_count\": " << warnings.size() << ",\n"
        << "  \"errors\": [";
    
    for (size_t i = 0; i < errors.size(); ++i) {
        if (i > 0) oss << ",";
        oss << "\n    \"" << errors[i] << "\"";
    }
    oss << "\n  ],\n"
        << "  \"warnings\": [";
    
    for (size_t i = 0; i < warnings.size(); ++i) {
        if (i > 0) oss << ",";
        oss << "\n    \"" << warnings[i] << "\"";
    }
    oss << "\n  ]\n"
        << "}";
    
    return oss.str();
}

// Фабричный метод
inline std::unique_ptr<IDataValidator> IDataValidator::createDefault() {
    return std::make_unique<DataValidator>();
}

} // namespace Correlator

#endif // CORRELATOR_DATA_VALIDATOR_HPP

