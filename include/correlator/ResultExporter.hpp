#ifndef CORRELATOR_RESULT_EXPORTER_HPP
#define CORRELATOR_RESULT_EXPORTER_HPP

#include "IResultExporter.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <filesystem>

namespace Correlator {

/**
 * @class ResultExporter
 * @brief Реализация экспорта результатов в JSON
 * 
 * Сохраняет промежуточные данные и результаты валидации
 * в JSON файлы для анализа и верификации.
 */
class ResultExporter : public IResultExporter {
private:
    std::string export_path_;
    std::string timestamp_;
    std::string timestamp_dir_;

    std::string getCurrentTimestamp() const {
        auto now = std::time(nullptr);
        struct tm timeinfo;
        #if defined(_WIN32) || defined(_WIN64)
            localtime_s(&timeinfo, &now);
        #else
            localtime_r(&now, &timeinfo);
        #endif
        
        char buffer[100];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S", &timeinfo);
        return std::string(buffer);
    }

    std::string getJSONFilename(const std::string& step_name) const {
        // Формат: Validation/YYYY-MM-DD_HH-MM-SS/StepN.json
        return timestamp_dir_ + "/" + step_name + ".json";
    }

    void ensureDirectoryExists(const std::string& path) const {
        std::filesystem::path dir(path);
        if (!dir.empty() && !std::filesystem::exists(dir)) {
            std::filesystem::create_directories(dir);
        }
    }

public:
    ResultExporter() : export_path_("Report/Validation"), timestamp_(getCurrentTimestamp()) {
        // Создать каталог с timestamp: Validation/YYYY-MM-DD_HH-MM-SS
        timestamp_dir_ = export_path_ + "/" + timestamp_;
        ensureDirectoryExists(timestamp_dir_);
        ensureDirectoryExists(export_path_);
    }

    void setExportPath(const std::string& path) override {
        export_path_ = path;
        // Пересоздать каталог с timestamp в новом пути
        timestamp_dir_ = export_path_ + "/" + timestamp_;
        ensureDirectoryExists(timestamp_dir_);
        ensureDirectoryExists(export_path_);
    }

    std::string getExportPath() const override {
        return export_path_;
    }

    void exportStep0(const std::vector<int32_t>& reference_signal,
                    const std::vector<int32_t>& input_signals,
                    const IConfiguration& config) override {
        std::string filename = getJSONFilename("Step0");
        std::ofstream file(filename);
        
        if (!file.is_open()) {
            return;
        }

        // Получить текущий timestamp для Step0
        auto now = std::time(nullptr);
        struct tm timeinfo;
        #if defined(_WIN32) || defined(_WIN64)
            localtime_s(&timeinfo, &now);
        #else
            localtime_r(&now, &timeinfo);
        #endif
        char timestamp_buffer[100];
        std::strftime(timestamp_buffer, sizeof(timestamp_buffer), "%Y-%m-%d %H:%M:%S", &timeinfo);

        file << "{\n"
             << "  \"step\": \"STEP0_M_SEQUENCE\",\n"
             << "  \"timestamp\": \"" << timestamp_buffer << "\",\n"
             << "  \"configuration\": " << config.toJSON() << ",\n"
             << "  \"reference_signal\": [";

        // Экспорт reference_signal
        for (size_t i = 0; i < reference_signal.size(); ++i) {
            file << reference_signal[i];
            if (i < reference_signal.size() - 1) {
                file << ", ";
            }
        }

        file << "],\n"
             << "  \"input_signals\": [\n";

        // Экспорт input_signals (разделены на отдельные сигналы)
        size_t fft_size = config.getFFTSize();
        int num_signals = config.getNumSignals();
        
        for (int sig_idx = 0; sig_idx < num_signals; ++sig_idx) {
            file << "    [";
            for (size_t i = 0; i < fft_size; ++i) {
                size_t idx = sig_idx * fft_size + i;
                if (idx < input_signals.size()) {
                    file << input_signals[idx];
                }
                if (i < fft_size - 1 && idx < input_signals.size() - 1) {
                    file << ", ";
                }
            }
            file << "]";
            if (sig_idx < num_signals - 1) {
                file << ",";
            }
            file << "\n";
        }

        file << "  ]\n"
             << "}";
        
        file.close();
    }

    void exportStep1(const IDataSnapshot& snapshot,
                    const IConfiguration& config,
                    const ValidationResult& validation) override {
        std::string filename = getJSONFilename("Step1");
        std::ofstream file(filename);
        
        if (!file.is_open()) {
            return;
        }

        file << "{\n"
             << "  \"step\": \"STEP1_REFERENCE_FFT\",\n"
             << "  \"timestamp\": \"" << snapshot.getTimestamp() << "\",\n"
             << "  \"configuration\": " << config.toJSON() << ",\n"
             << "  \"data\": " << snapshot.toJSON(IDataSnapshot::Step::STEP1_REFERENCE_FFT) << ",\n"
             << "  \"validation\": " << validation.toJSON() << "\n"
             << "}";
        
        file.close();
    }

    void exportStep2(const IDataSnapshot& snapshot,
                    const IConfiguration& config,
                    const ValidationResult& validation) override {
        std::string filename = getJSONFilename("Step2");
        std::ofstream file(filename);
        
        if (!file.is_open()) {
            return;
        }

        file << "{\n"
             << "  \"step\": \"STEP2_INPUT_FFT\",\n"
             << "  \"timestamp\": \"" << snapshot.getTimestamp() << "\",\n"
             << "  \"configuration\": " << config.toJSON() << ",\n"
             << "  \"data\": " << snapshot.toJSON(IDataSnapshot::Step::STEP2_INPUT_FFT) << ",\n"
             << "  \"validation\": " << validation.toJSON() << "\n"
             << "}";
        
        file.close();
    }

    void exportStep3(const IDataSnapshot& snapshot,
                    const IConfiguration& config,
                    const ValidationResult& validation) override {
        std::string filename = getJSONFilename("Step3");
        std::ofstream file(filename);
        
        if (!file.is_open()) {
            return;
        }

        file << "{\n"
             << "  \"step\": \"STEP3_CORRELATION\",\n"
             << "  \"timestamp\": \"" << snapshot.getTimestamp() << "\",\n"
             << "  \"configuration\": " << config.toJSON() << ",\n"
             << "  \"data\": " << snapshot.toJSON(IDataSnapshot::Step::STEP3_PEAKS) << ",\n"
             << "  \"validation\": " << validation.toJSON() << "\n"
             << "}";
        
        file.close();
    }

    void exportFinalReport(const IDataSnapshot& snapshot,
                          const IConfiguration& config) override {
        std::string filename = export_path_ + "/final_report_" + timestamp_ + ".json";
        std::ofstream file(filename);
        
        if (!file.is_open()) {
            return;
        }

        file << "{\n"
             << "  \"report_type\": \"FINAL_REPORT\",\n"
             << "  \"timestamp\": \"" << snapshot.getTimestamp() << "\",\n"
             << "  \"configuration\": " << config.toJSON() << ",\n"
             << "  \"statistics\": \"" << snapshot.getStatistics() << "\",\n"
             << "  \"all_steps\": {\n"
             << "    \"step1\": " << snapshot.toJSON(IDataSnapshot::Step::STEP1_REFERENCE_FFT) << ",\n"
             << "    \"step2\": " << snapshot.toJSON(IDataSnapshot::Step::STEP2_INPUT_FFT) << ",\n"
             << "    \"step3\": " << snapshot.toJSON(IDataSnapshot::Step::STEP3_PEAKS) << "\n"
             << "  }\n"
             << "}";
        
        file.close();
    }
};

// Фабричный метод
inline std::unique_ptr<IResultExporter> IResultExporter::createDefault() {
    return std::make_unique<ResultExporter>();
}

} // namespace Correlator

#endif // CORRELATOR_RESULT_EXPORTER_HPP

