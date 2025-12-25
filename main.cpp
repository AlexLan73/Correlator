#include "include/correlator/Correlator.hpp"
#include "include/correlator/OpenCLFFTBackend.hpp"
#include "include/profiler.hpp"
#include "include/debug_log.hpp"
#include <iostream>
#include <sstream>
#include <vector>
#include <cstdint>
#include <map>
#include <string>
#include <cstdio>
#include "src/Test_GPU_and_OpenCl.h"

using namespace Correlator;

// Генератор M-sequence
std::vector<int32_t> generateMSequence(size_t length, uint32_t seed = 0x1) {
    std::vector<int32_t> sequence(length);
    uint32_t lfsr = seed;
    const uint32_t POLY = 0xB8000000;

    for (size_t i = 0; i < length; i++) {
        int bit = (lfsr >> 31) & 1;
//        sequence[i] = bit ? 10000 : -10000;
        sequence[i] = bit ? 1 : -1;
        if (bit) {
            lfsr = ((lfsr << 1) ^ POLY);
        } else {
            lfsr = (lfsr << 1);
        }
    }

    return sequence;
}

int main() {
    test_gpu::test_gpu_opencl();

    COUT_LOG("╔══════════════════════════════════════════════════════════════╗\n");
    COUT_LOG("║     FFT CORRELATOR - Пример использования архитектуры       ║\n");
    COUT_LOG("╚══════════════════════════════════════════════════════════════╝\n\n");

    // Инициализация профилировщика
    Profiler profiler;

    try {
        // 1. Создать конфигурацию
        COUT_LOG("[1] Создание конфигурации...\n");
        auto config = IConfiguration::createDefault();

        //  работало  _ = 1<<18;   config->setNumShifts(10);  // 32
        auto xx_ = 1<<15;
        config->setFFTSize(xx_);      // 2^19
        config->setNumShifts(10);  // 32
        config->setNumSignals(5); // 5
        config->setNumOutputPoints(2000);

         config->setScaleFactor(1.0f / 32768.0f);

        if (!config->validate()) {
            std::cerr << "Ошибка валидации конфигурации: " << config->getValidationErrors() << "\n";
            return 1;
        }
        COUT_LOG("✓ Конфигурация создана и валидирована\n\n");

        // 2. Создать бэкенд (OpenCL)
        COUT_LOG("[2] Создание OpenCL бэкенда...\n");
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
        COUT_LOG("✓ Бэкенд создан\n\n");

        // 3. Сохранить значения конфигурации перед созданием pipeline
        size_t fft_size = config->getFFTSize();
        int num_shifts = config->getNumShifts();
        int num_signals = config->getNumSignals();
        int num_output_points = config->getNumOutputPoints();
        
        // 4. Подготовить данные (используем значения из конфигурации)
        COUT_LOG("[4] Генерация тестовых данных...\n");
        std::vector<int32_t> reference_signal = generateMSequence(fft_size);

        std::vector<int32_t> input_signals;
        input_signals.reserve(num_signals * fft_size);
        for (int i = 0; i < num_signals; ++i) {
            auto signal = generateMSequence(fft_size, 0x1 + i);
            input_signals.insert(input_signals.end(), signal.begin(), signal.end());
        }
        COUT_LOG("✓ Данные сгенерированы\n\n");

        // 4.5. Создать exporter для экспорта Step0 (и использования в pipeline)
        COUT_LOG("[4.5] Создание exporter...\n");
        auto exporter = IResultExporter::createDefault();
        
        // Экспорт M-последовательности (Step0)
        exporter->exportStep0(reference_signal, input_signals, *config);
        COUT_LOG("✓ Step0 данные экспортированы\n\n");

        // 5. Создать pipeline
        COUT_LOG("[5] Создание CorrelationPipeline...\n");
        CorrelationPipeline pipeline(std::move(backend), std::move(config));
        
        // Установить тот же exporter в pipeline для использования одного timestamp каталога
        pipeline.setExporter(std::move(exporter));
        const auto& config_ref = pipeline.getConfiguration();
        COUT_LOG("✓ Pipeline создан\n\n");

        // 6. Выполнить весь pipeline с профилированием каждого шага
        COUT_LOG("[6] Выполнение полного pipeline...\n");
        COUT_LOG("   Step 1: Reference FFT\n");
        COUT_LOG("   Step 2: Input FFT\n");
        COUT_LOG("   Step 3: Correlation\n\n");

        // Инициализация pipeline
        if (!pipeline.initialize()) {
            std::cerr << "Ошибка инициализации pipeline\n";
            return 1;
        }

        // Step 1 с профилированием
        profiler.start("Step1_Total");
        if (!pipeline.executeStep1(reference_signal, config_ref.getNumShifts())) {
            std::cerr << "Ошибка выполнения Step 1\n";
            return 1;
        }
        profiler.stop("Step1_Total", Profiler::MILLISECONDS);

        // Step 2 с профилированием
        profiler.start("Step2_Total");
        if (!pipeline.executeStep2(input_signals, config_ref.getNumSignals())) {
            std::cerr << "Ошибка выполнения Step 2\n";
            return 1;
        }
        profiler.stop("Step2_Total", Profiler::MILLISECONDS);

        // Step 3 с профилированием
        profiler.start("Step3_Total");
        if (!pipeline.executeStep3(config_ref.getNumSignals(), config_ref.getNumShifts(), 
                                   config_ref.getNumOutputPoints())) {
            std::cerr << "Ошибка выполнения Step 3\n";
            return 1;
        }
        profiler.stop("Step3_Total", Profiler::MILLISECONDS);

        COUT_LOG("✓ Pipeline выполнен успешно\n\n");

        // 6. Получить результаты
        COUT_LOG("[6] Получение результатов...\n");
        const auto& snapshot = pipeline.getSnapshot();
        const auto& peaks = snapshot.getPeaks();

        {
            std::ostringstream oss;
            oss << "✓ Получено " << peaks.size() << " пиков\n";
            COUT_LOG(oss.str());
        }
        {
            std::ostringstream oss;
            oss << "   Формат: [" << config_ref.getNumSignals() << " сигналов]["
                << config_ref.getNumShifts() << " сдвигов][" 
                << config_ref.getNumOutputPoints() << " точек] = "
                << (config_ref.getNumSignals() * config_ref.getNumShifts() * config_ref.getNumOutputPoints()) 
                << " значений\n\n";
            COUT_LOG(oss.str());
        }

        // 7. Экспорт в JSON (уже выполнен автоматически на каждом этапе)
        COUT_LOG("[7] JSON файлы сохранены в Report/Validation/\n");
        COUT_LOG("   - validation_step1_*.json\n");
        COUT_LOG("   - validation_step2_*.json\n");
        COUT_LOG("   - validation_step3_*.json\n");
        COUT_LOG("   - final_report_*.json\n\n");

        // 8. Получить информацию о GPU
        COUT_LOG("[8] Информация о GPU:\n");
        const auto& backend_info = pipeline.getBackend();
        {
            std::ostringstream oss;
            oss << "   Платформа: " << backend_info.getPlatformName() << "\n"
                << "   Устройство: " << backend_info.getDeviceName() << "\n"
                << "   Драйвер: " << backend_info.getDriverVersion() << "\n"
                << "   API: " << backend_info.getAPIVersion() << "\n\n";
            COUT_LOG(oss.str());
        }

        COUT_LOG("═══════════════════════════════════════════════════════════\n");
        COUT_LOG("✨ ВСЕ ЭТАПЫ ВЫПОЛНЕНЫ УСПЕШНО! ✨\n");
        COUT_LOG("═══════════════════════════════════════════════════════════\n\n");

        // Экспорт и вывод результатов профилирования
        COUT_LOG("[PROFILER] Экспорт отчета профилирования...\n");
        
        // Получить OperationTiming данные из pipeline
        OperationTiming step1_upload, step1_fft;
        OperationTiming step2_upload, step2_fft;
        OperationTiming step3_copy, step3_ifft, step3_download;
        
        pipeline.getStep1Timings(step1_upload, step1_fft);
        pipeline.getStep2Timings(step2_upload, step2_fft);
        pipeline.getStep3Timings(step3_copy, step3_ifft, step3_download);
        
        // Сформировать step_details аналогично CorrelatorW
        std::map<std::string, std::map<std::string, double>> step_details;
        
        // Step 1 детали
        char step1_label[64];
        std::snprintf(step1_label, sizeof(step1_label), "FFT (%d) total GPU time", config_ref.getNumShifts());
        char step1_label_cpu[64];
        std::snprintf(step1_label_cpu, sizeof(step1_label_cpu), "FFT (%d) CPU wait", config_ref.getNumShifts());
        step_details["Step1"]["Upload total GPU time"] = step1_upload.total_gpu_ms;
        step_details["Step1"]["Upload CPU wait"] = step1_upload.cpu_wait_ms;
        step_details["Step1"][step1_label] = step1_fft.total_gpu_ms;
        step_details["Step1"][step1_label_cpu] = step1_fft.cpu_wait_ms;
        
        // Step 2 детали
        char step2_label[64];
        std::snprintf(step2_label, sizeof(step2_label), "FFT (%d) total GPU time", config_ref.getNumSignals());
        char step2_label_cpu[64];
        std::snprintf(step2_label_cpu, sizeof(step2_label_cpu), "FFT (%d) CPU wait", config_ref.getNumSignals());
        step_details["Step2"]["Upload total GPU time"] = step2_upload.total_gpu_ms;
        step_details["Step2"]["Upload CPU wait"] = step2_upload.cpu_wait_ms;
        step_details["Step2"][step2_label] = step2_fft.total_gpu_ms;
        step_details["Step2"][step2_label_cpu] = step2_fft.cpu_wait_ms;
        
        // Step 3 детали
        int total_correlations = config_ref.getNumSignals() * config_ref.getNumShifts();
        char step3_label[64];
        std::snprintf(step3_label, sizeof(step3_label), "Inverse FFT (%d) total GPU time", total_correlations);
        char step3_label_cpu[64];
        std::snprintf(step3_label_cpu, sizeof(step3_label_cpu), "Inverse FFT (%d) CPU wait", total_correlations);
        step_details["Step3"]["Complex multiply total GPU time"] = step3_copy.total_gpu_ms;
        step_details["Step3"]["Complex multiply CPU wait"] = step3_copy.cpu_wait_ms;
        step_details["Step3"][step3_label] = step3_ifft.total_gpu_ms;
        step_details["Step3"][step3_label_cpu] = step3_ifft.cpu_wait_ms;
        step_details["Step3"]["Download total GPU time"] = step3_download.total_gpu_ms;
        step_details["Step3"]["Download CPU wait"] = step3_download.cpu_wait_ms;
        
        // Получить информацию о GPU из backend
        cl_device_id device_id = backend_info.getDeviceId();
        Profiler::GPUInfo gpu_info = Profiler::get_gpu_info(device_id);

        // Параметры конфигурации для отчета
        Profiler::ConfigParams config_params;
        config_params.fft_size = config_ref.getFFTSize();
        config_params.num_shifts = config_ref.getNumShifts();
        config_params.num_signals = config_ref.getNumSignals();
        config_params.num_output_points = config_ref.getNumOutputPoints();

        // Экспорт в Markdown
        if (profiler.export_to_markdown("Report/profiling_report.md", step_details, gpu_info, config_params)) {
            COUT_LOG("✓ Отчет профилирования сохранен: Report/profiling_report.md\n");
        } else {
            COUT_LOG("⚠️ Не удалось сохранить отчет профилирования\n");
        }

        // Экспорт в JSON
        if (profiler.export_to_json("Report/profiling_report.json", step_details, gpu_info)) {
            COUT_LOG("✓ JSON отчет профилирования сохранен: Report/JSON/profiling_report.json\n");
        } else {
            COUT_LOG("⚠️ Не удалось сохранить JSON отчет профилирования\n");
        }

        COUT_LOG("[PROFILER] Результаты профилирования:\n");
        profiler.print_all("FFT CORRELATOR PROFILING RESULTS");

        COUT_LOG("✓ Профилирование завершено\n");

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << "\n";
        return 1;
    }
}
