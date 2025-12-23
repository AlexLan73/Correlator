#ifndef ICORRELATOR_CONFIGURATION_HPP
#define ICORRELATOR_CONFIGURATION_HPP

#include <cstdint>
#include <string>
#include <memory>

namespace Correlator {

/**
 * @class IConfiguration
 * @brief Интерфейс конфигурации коррелятора
 * 
 * Инкапсулирует все параметры конфигурации для FFT коррелятора.
 * Поддерживает валидацию и сериализацию в JSON.
 */
class IConfiguration {
public:
    virtual ~IConfiguration() = default;

    // Основные параметры
    virtual size_t getFFTSize() const = 0;
    virtual int getNumShifts() const = 0;
    virtual int getNumSignals() const = 0;
    virtual int getNumOutputPoints() const = 0;
    virtual float getScaleFactor() const = 0;

    // Установка параметров
    virtual void setFFTSize(size_t size) = 0;
    virtual void setNumShifts(int shifts) = 0;
    virtual void setNumSignals(int signals) = 0;
    virtual void setNumOutputPoints(int points) = 0;
    virtual void setScaleFactor(float factor) = 0;

    // Валидация
    virtual bool validate() const = 0;
    virtual std::string getValidationErrors() const = 0;

    // Сериализация
    virtual std::string toJSON() const = 0;
    virtual bool fromJSON(const std::string& json) = 0;

    // Фабричный метод
    static std::unique_ptr<IConfiguration> createDefault();
};

} // namespace Correlator

#endif // ICORRELATOR_CONFIGURATION_HPP

