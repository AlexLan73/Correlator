#ifndef ICORRELATOR_RESULT_EXPORTER_HPP
#define ICORRELATOR_RESULT_EXPORTER_HPP

#include "IDataSnapshot.hpp"
#include "IConfiguration.hpp"
#include "IDataValidator.hpp"
#include <string>
#include <memory>
#include <vector>
#include <cstdint>

namespace Correlator {

/**
 * @class IResultExporter
 * @brief Интерфейс для экспорта результатов в JSON
 * 
 * Экспортирует промежуточные данные и результаты валидации
 * на каждом этапе для верификации алгоритма.
 */
class IResultExporter {
public:
    virtual ~IResultExporter() = default;

    // Экспорт на каждом этапе
    virtual void exportStep0(const std::vector<int32_t>& reference_signal,
                            const std::vector<int32_t>& input_signals,
                            const IConfiguration& config) = 0;
    
    virtual void exportStep1(const IDataSnapshot& snapshot,
                            const IConfiguration& config,
                            const ValidationResult& validation) = 0;
    
    virtual void exportStep2(const IDataSnapshot& snapshot,
                            const IConfiguration& config,
                            const ValidationResult& validation) = 0;
    
    virtual void exportStep3(const IDataSnapshot& snapshot,
                            const IConfiguration& config,
                            const ValidationResult& validation) = 0;

    // Финальный отчет
    virtual void exportFinalReport(const IDataSnapshot& snapshot,
                                  const IConfiguration& config) = 0;

    // Настройка пути экспорта
    virtual void setExportPath(const std::string& path) = 0;
    virtual std::string getExportPath() const = 0;

    // Фабричный метод
    static std::unique_ptr<IResultExporter> createDefault();
};

} // namespace Correlator

#endif // ICORRELATOR_RESULT_EXPORTER_HPP

