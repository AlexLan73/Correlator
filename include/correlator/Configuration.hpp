#ifndef CORRELATOR_CONFIGURATION_HPP
#define CORRELATOR_CONFIGURATION_HPP

#include "IConfiguration.hpp"
#include <sstream>
#include <stdexcept>
#include <iomanip>

namespace Correlator {

/**
 * @class Configuration
 * @brief Реализация конфигурации коррелятора
 * 
 * Хранит и валидирует параметры конфигурации.
 * Поддерживает JSON сериализацию для сохранения/загрузки настроек.
 */
class Configuration : public IConfiguration {
private:
    size_t fft_size_;
    int num_shifts_;
    int num_signals_;
    int num_output_points_;
    float scale_factor_;

public:
    Configuration()
        : fft_size_(32768),      // 2^15
          num_shifts_(40),
          num_signals_(50),
          num_output_points_(5),
          scale_factor_(1.0f / 32768.0f) {}

    Configuration(size_t fft_size, int num_shifts, int num_signals, 
                  int num_output_points, float scale_factor)
        : fft_size_(fft_size),
          num_shifts_(num_shifts),
          num_signals_(num_signals),
          num_output_points_(num_output_points),
          scale_factor_(scale_factor) {
        if (!validate()) {
            throw std::invalid_argument("Invalid configuration: " + getValidationErrors());
        }
    }

    // Getters
    size_t getFFTSize() const override { return fft_size_; }
    int getNumShifts() const override { return num_shifts_; }
    int getNumSignals() const override { return num_signals_; }
    int getNumOutputPoints() const override { return num_output_points_; }
    float getScaleFactor() const override { return scale_factor_; }

    // Setters
    void setFFTSize(size_t size) override { fft_size_ = size; }
    void setNumShifts(int shifts) override { num_shifts_ = shifts; }
    void setNumSignals(int signals) override { num_signals_ = signals; }
    void setNumOutputPoints(int points) override { num_output_points_ = points; }
    void setScaleFactor(float factor) override { scale_factor_ = factor; }

    // Валидация
    bool validate() const override {
        return fft_size_ > 0 && 
               num_shifts_ > 0 && 
               num_signals_ > 0 && 
               num_output_points_ > 0 && 
               scale_factor_ > 0.0f;
    }

    std::string getValidationErrors() const override {
        std::ostringstream oss;
        bool has_errors = false;

        if (fft_size_ == 0) {
            oss << "FFT size must be > 0; ";
            has_errors = true;
        }
        if (num_shifts_ <= 0) {
            oss << "Number of shifts must be > 0; ";
            has_errors = true;
        }
        if (num_signals_ <= 0) {
            oss << "Number of signals must be > 0; ";
            has_errors = true;
        }
        if (num_output_points_ <= 0) {
            oss << "Number of output points must be > 0; ";
            has_errors = true;
        }
        if (scale_factor_ <= 0.0f) {
            oss << "Scale factor must be > 0; ";
            has_errors = true;
        }

        return has_errors ? oss.str() : "";
    }

    // JSON сериализация
    std::string toJSON() const override {
        std::ostringstream oss;
        oss << "{\n"
            << "  \"fft_size\": " << fft_size_ << ",\n"
            << "  \"num_shifts\": " << num_shifts_ << ",\n"
            << "  \"num_signals\": " << num_signals_ << ",\n"
            << "  \"num_output_points\": " << num_output_points_ << ",\n"
            << "  \"scale_factor\": " << std::fixed << std::setprecision(9) << scale_factor_ << "\n"
            << "}";
        return oss.str();
    }

    bool fromJSON(const std::string& json) override {
        // Простая реализация (можно улучшить с помощью библиотеки JSON)
        // Для полной реализации лучше использовать nlohmann/json или rapidjson
        // Здесь упрощенная версия для демонстрации
        try {
            // TODO: Реализовать полноценный парсинг JSON
            // Пока возвращаем false, так как нужна библиотека
            return false;
        } catch (...) {
            return false;
        }
    }
};

// Фабричный метод
inline std::unique_ptr<IConfiguration> IConfiguration::createDefault() {
    return std::make_unique<Configuration>();
}

} // namespace Correlator

#endif // CORRELATOR_CONFIGURATION_HPP

