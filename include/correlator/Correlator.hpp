#ifndef CORRELATOR_HPP
#define CORRELATOR_HPP

/**
 * @file Correlator.hpp
 * @brief Главный заголовочный файл для FFT Correlator
 * 
 * Профессиональная архитектура классов для FFT коррелятора
 * с поддержкой валидации, JSON экспорта и расширяемости.
 * 
 * @author Senior Architecture Team
 * @date 2025-12-21
 */

// Основные интерфейсы
#include "IConfiguration.hpp"
#include "IDataSnapshot.hpp"
#include "IFFTBackend.hpp"
#include "IDataValidator.hpp"
#include "IResultExporter.hpp"

// Реализации
#include "Configuration.hpp"
#include "DataSnapshot.hpp"
#include "DataValidator.hpp"
#include "ResultExporter.hpp"
#include "CorrelationPipeline.hpp"

/**
 * @namespace Correlator
 * @brief Пространство имен для всех классов коррелятора
 * 
 * Все классы коррелятора находятся в этом namespace для
 * избежания конфликтов имен и лучшей организации кода.
 */
namespace Correlator {

/**
 * @brief Быстрый старт: создание и использование pipeline
 * 
 * Пример использования:
 * @code
 * // Создать конфигурацию
 * auto config = IConfiguration::createDefault();
 * 
 * // Создать бэкенд (OpenCL)
 * auto backend = IFFTBackend::createOpenCLBackend();
 * 
 * // Создать pipeline
 * CorrelationPipeline pipeline(std::move(backend), std::move(config));
 * 
 * // Выполнить весь pipeline
 * std::vector<int32_t> reference_signal = generateMSequence(...);
 * std::vector<int32_t> input_signals = generateMSequences(...);
 * 
 * if (pipeline.executeFullPipeline(reference_signal, input_signals)) {
 *     // Получить результаты
 *     const auto& snapshot = pipeline.getSnapshot();
 *     const auto& peaks = snapshot.getPeaks();
 *     // ...
 * }
 * @endcode
 */

} // namespace Correlator

#endif // CORRELATOR_HPP

