#ifndef CORRELATOR_PIPELINE_HPP
#define CORRELATOR_PIPELINE_HPP

#include "IFFTBackend.hpp"
#include "IConfiguration.hpp"
#include "IDataSnapshot.hpp"
#include "IDataValidator.hpp"
#include "IResultExporter.hpp"
#include <memory>
#include <vector>
#include <string>

namespace Correlator {

/**
 * @class CorrelationPipeline
 * @brief Главный класс для оркестрации всего pipeline корреляции
 * 
 * Управляет выполнением всех этапов обработки:
 * - Step 1: Обработка опорных сигналов
 * - Step 2: Обработка входных сигналов  
 * - Step 3: Корреляция
 * 
 * Автоматически сохраняет промежуточные данные, валидирует результаты,
 * экспортирует данные в JSON для верификации.
 */
class CorrelationPipeline {
private:
    std::unique_ptr<IFFTBackend> backend_;
    std::unique_ptr<IConfiguration> config_;
    std::unique_ptr<IDataSnapshot> snapshot_;
    std::unique_ptr<IDataValidator> validator_;
    std::unique_ptr<IResultExporter> exporter_;

    bool step1_completed_;
    bool step2_completed_;
    bool step3_completed_;

    // Хранение данных профилирования для каждого шага
    OperationTiming step1_upload_timing_;
    OperationTiming step1_fft_timing_;
    OperationTiming step2_upload_timing_;
    OperationTiming step2_fft_timing_;
    OperationTiming step3_copy_timing_;
    OperationTiming step3_ifft_timing_;
    OperationTiming step3_download_timing_;

public:
    /**
     * @brief Конструктор
     * @param backend FFT бэкенд (OpenCL, CUDA, etc.)
     * @param config Конфигурация
     */
    CorrelationPipeline(
        std::unique_ptr<IFFTBackend> backend,
        std::unique_ptr<IConfiguration> config
    ) : backend_(std::move(backend)),
        config_(std::move(config)),
        snapshot_(std::make_unique<DataSnapshot>()),
        validator_(IDataValidator::createDefault()),
        exporter_(IResultExporter::createDefault()),
        step1_completed_(false),
        step2_completed_(false),
        step3_completed_(false) {
        
        if (!config_->validate()) {
            throw std::invalid_argument("Invalid configuration: " + 
                                      config_->getValidationErrors());
        }
    }

    /**
     * @brief Инициализация pipeline
     */
    bool initialize() {
        if (!backend_->initialize()) {
            return false;
        }
        return true;
    }

    /**
     * @brief Step 1: Обработка опорных сигналов
     * @param reference_signal Опорный сигнал (M-sequence)
     * @param num_shifts Количество циклических сдвигов
     */
    bool executeStep1(const std::vector<int32_t>& reference_signal, int num_shifts) {
        if (step1_completed_) {
            // Уже выполнено, можно пропустить
            return true;
        }

        OperationTiming upload_timing, fft_timing;
        
        if (!backend_->step1_ProcessReferenceSignals(reference_signal, num_shifts, 
                                                      upload_timing, fft_timing)) {
            return false;
        }

        // Сохранить данные профилирования
        step1_upload_timing_ = upload_timing;
        step1_fft_timing_ = fft_timing;

        // Получить результаты и сохранить в snapshot
        std::vector<ComplexFloat> reference_fft;
        if (!backend_->getReferenceFFT(reference_fft)) {
            return false;
        }

        snapshot_->saveReferenceFFT(reference_fft, num_shifts, config_->getFFTSize());

        // Валидация
        auto validation = validator_->validateStep1(*snapshot_, *config_);
        if (!validation.is_valid) {
            // Логируем ошибки, но не останавливаем выполнение
            // (можно добавить опцию strict validation)
        }

        // Экспорт в JSON
        exporter_->exportStep1(*snapshot_, *config_, validation);

        step1_completed_ = true;
        return true;
    }

    /**
     * @brief Step 2: Обработка входных сигналов
     * @param input_signals Входные сигналы (50 × M-sequence)
     * @param num_signals Количество входных сигналов
     */
    bool executeStep2(const std::vector<int32_t>& input_signals, int num_signals) {
        if (!step1_completed_) {
            throw std::runtime_error("Step 1 must be completed before Step 2");
        }

        if (step2_completed_) {
            return true;
        }

        OperationTiming upload_timing, fft_timing;
        
        if (!backend_->step2_ProcessInputSignals(input_signals, num_signals, 
                                                 upload_timing, fft_timing)) {
            return false;
        }

        // Сохранить данные профилирования
        step2_upload_timing_ = upload_timing;
        step2_fft_timing_ = fft_timing;

        // Получить результаты и сохранить в snapshot
        std::vector<ComplexFloat> input_fft;
        if (!backend_->getInputFFT(input_fft)) {
            return false;
        }

        snapshot_->saveInputFFT(input_fft, num_signals, config_->getFFTSize());

        // Валидация
        auto validation = validator_->validateStep2(*snapshot_, *config_);
        if (!validation.is_valid) {
            // Логируем ошибки
        }

        // Экспорт в JSON
        exporter_->exportStep2(*snapshot_, *config_, validation);

        step2_completed_ = true;
        return true;
    }

    /**
     * @brief Step 3: Корреляция
     * @param num_signals Количество входных сигналов
     * @param num_shifts Количество сдвигов
     * @param n_kg Количество выходных точек
     */
    bool executeStep3(int num_signals, int num_shifts, int n_kg) {
        if (!step1_completed_ || !step2_completed_) {
            throw std::runtime_error("Step 1 and Step 2 must be completed before Step 3");
        }

        if (step3_completed_) {
            return true;
        }

        OperationTiming copy_timing, ifft_timing, download_timing;
        
        if (!backend_->step3_ComputeCorrelation(num_signals, num_shifts, n_kg,
                                               copy_timing, ifft_timing, download_timing)) {
            return false;
        }

        // Сохранить данные профилирования
        step3_copy_timing_ = copy_timing;
        step3_ifft_timing_ = ifft_timing;
        step3_download_timing_ = download_timing;

        // Получить результаты и сохранить в snapshot
        std::vector<float> peaks;
        if (!backend_->getCorrelationPeaks(peaks)) {
            return false;
        }

        snapshot_->savePeaks(peaks, num_signals, num_shifts, n_kg);

        // Валидация
        auto validation = validator_->validateStep3(*snapshot_, *config_);
        if (!validation.is_valid) {
            // Логируем ошибки
        }

        // Экспорт в JSON
        exporter_->exportStep3(*snapshot_, *config_, validation);

        step3_completed_ = true;
        return true;
    }

    /**
     * @brief Выполнить весь pipeline
     */
    bool executeFullPipeline(
        const std::vector<int32_t>& reference_signal,
        const std::vector<int32_t>& input_signals
    ) {
        if (!initialize()) {
            return false;
        }

        if (!executeStep1(reference_signal, config_->getNumShifts())) {
            return false;
        }

        if (!executeStep2(input_signals, config_->getNumSignals())) {
            return false;
        }

        if (!executeStep3(config_->getNumSignals(), config_->getNumShifts(), 
                         config_->getNumOutputPoints())) {
            return false;
        }

        // Экспорт финального отчета
        exporter_->exportFinalReport(*snapshot_, *config_);

        return true;
    }

    // Getters
    const IDataSnapshot& getSnapshot() const { return *snapshot_; }
    IDataSnapshot& getSnapshot() { return *snapshot_; }
    const IConfiguration& getConfiguration() const { return *config_; }
    const IFFTBackend& getBackend() const { return *backend_; }
    IFFTBackend& getBackend() { return *backend_; }

    // Установка exporter (для использования одного и того же timestamp каталога)
    void setExporter(std::unique_ptr<IResultExporter> exporter) {
        exporter_ = std::move(exporter);
    }

    // Получение данных профилирования
    void getStep1Timings(OperationTiming& upload, OperationTiming& fft) const {
        upload = step1_upload_timing_;
        fft = step1_fft_timing_;
    }

    void getStep2Timings(OperationTiming& upload, OperationTiming& fft) const {
        upload = step2_upload_timing_;
        fft = step2_fft_timing_;
    }

    void getStep3Timings(OperationTiming& copy, OperationTiming& ifft, OperationTiming& download) const {
        copy = step3_copy_timing_;
        ifft = step3_ifft_timing_;
        download = step3_download_timing_;
    }

    // Очистка
    void cleanup() {
        if (backend_) {
            backend_->cleanup();
        }
    }

    ~CorrelationPipeline() {
        cleanup();
    }
};

} // namespace Correlator

#endif // CORRELATOR_PIPELINE_HPP

