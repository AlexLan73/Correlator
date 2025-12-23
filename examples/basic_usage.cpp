/**
 * @file basic_usage.cpp
 * @brief Пример использования новой архитектуры FFT Correlator
 * 
 * Демонстрирует использование CorrelationPipeline с валидацией
 * и JSON экспортом на каждом этапе.
 */

#include "../include/correlator/Correlator.hpp"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>

using namespace Correlator;

// Генератор M-sequence
std::vector<int32_t> generateMSequence(size_t length, uint32_t seed = 0x1) {
    std::vector<int32_t> sequence(length);
    uint32_t lfsr = seed;
    const uint32_t POLY = 0xB8000000;

    for (size_t i = 0; i < length; i++) {
        int bit = (lfsr >> 31) & 1;
        sequence[i] = bit ? 10000 : -10000;
        if (bit) {
            lfsr = ((lfsr << 1) ^ POLY);
        } else {
            lfsr = (lfsr << 1);
        }
    }

    return sequence;
}

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║     FFT CORRELATOR - Пример использования архитектуры       ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    try {
        // 1. Создать конфигурацию
        std::cout << "[1] Создание конфигурации...\n";
        auto config = IConfiguration::createDefault();
        config->setFFTSize(32768);      // 2^15
        config->setNumShifts(40);
        config->setNumSignals(50);
        config->setNumOutputPoints(5);
        config->setScaleFactor(1.0f / 32768.0f);

        if (!config->validate()) {
            std::cerr << "Ошибка валидации конфигурации: " << config->getValidationErrors() << "\n";
            return 1;
        }
        std::cout << "✓ Конфигурация создана и валидирована\n\n";

        // 2. Создать бэкенд (OpenCL)
        std::cout << "[2] Создание OpenCL бэкенда...\n";
        auto backend = IFFTBackend::createOpenCLBackend();
        
        // Установить конфигурацию в бэкенд
        auto opencl_backend = dynamic_cast<OpenCLFFTBackend*>(backend.get());
        if (opencl_backend) {
            opencl_backend->setConfiguration(
                config->getFFTSize(),
                config->getNumShifts(),
                config->getNumSignals(),
                config->getNumOutputPoints(),
                config->getScaleFactor()
            );
        }
        std::cout << "✓ Бэкенд создан\n\n";

        // 3. Создать pipeline
        std::cout << "[3] Создание CorrelationPipeline...\n";
        CorrelationPipeline pipeline(std::move(backend), std::move(config));
        std::cout << "✓ Pipeline создан\n\n";

        // 4. Подготовить данные
        std::cout << "[4] Генерация тестовых данных...\n";
        std::vector<int32_t> reference_signal = generateMSequence(32768);
        
        std::vector<int32_t> input_signals;
        input_signals.reserve(50 * 32768);
        for (int i = 0; i < 50; ++i) {
            auto signal = generateMSequence(32768, 0x1 + i);
            input_signals.insert(input_signals.end(), signal.begin(), signal.end());
        }
        std::cout << "✓ Данные сгенерированы\n\n";

        // 5. Выполнить весь pipeline
        std::cout << "[5] Выполнение полного pipeline...\n";
        std::cout << "   Step 1: Reference FFT\n";
        std::cout << "   Step 2: Input FFT\n";
        std::cout << "   Step 3: Correlation\n\n";

        if (!pipeline.executeFullPipeline(reference_signal, input_signals)) {
            std::cerr << "Ошибка выполнения pipeline\n";
            return 1;
        }

        std::cout << "✓ Pipeline выполнен успешно\n\n";

        // 6. Получить результаты
        std::cout << "[6] Получение результатов...\n";
        const auto& snapshot = pipeline.getSnapshot();
        const auto& peaks = snapshot.getPeaks();
        
        std::cout << "✓ Получено " << peaks.size() << " пиков\n";
        std::cout << "   Формат: [50 сигналов][40 сдвигов][5 точек] = " 
                  << (50 * 40 * 5) << " значений\n\n";

        // 7. Экспорт в JSON (уже выполнен автоматически на каждом этапе)
        std::cout << "[7] JSON файлы сохранены в Report/Validation/\n";
        std::cout << "   - validation_step1_*.json\n";
        std::cout << "   - validation_step2_*.json\n";
        std::cout << "   - validation_step3_*.json\n";
        std::cout << "   - final_report_*.json\n\n";

        // 8. Получить информацию о GPU
        std::cout << "[8] Информация о GPU:\n";
        const auto& backend_info = pipeline.getBackend();
        std::cout << "   Платформа: " << backend_info.getPlatformName() << "\n";
        std::cout << "   Устройство: " << backend_info.getDeviceName() << "\n";
        std::cout << "   Драйвер: " << backend_info.getDriverVersion() << "\n";
        std::cout << "   API: " << backend_info.getAPIVersion() << "\n\n";

        std::cout << "═══════════════════════════════════════════════════════════\n";
        std::cout << "✨ ВСЕ ЭТАПЫ ВЫПОЛНЕНЫ УСПЕШНО! ✨\n";
        std::cout << "═══════════════════════════════════════════════════════════\n\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << "\n";
        return 1;
    }
}

