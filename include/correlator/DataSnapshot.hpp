#ifndef CORRELATOR_DATA_SNAPSHOT_HPP
#define CORRELATOR_DATA_SNAPSHOT_HPP

#include "IDataSnapshot.hpp"
#include <ctime>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace Correlator {

/**
 * @class DataSnapshot
 * @brief Реализация сохранения промежуточных данных
 * 
 * Сохраняет данные на каждом этапе для валидации и верификации.
 * Поддерживает экспорт в JSON и статистический анализ.
 */
class DataSnapshot : public IDataSnapshot {
private:
    std::vector<ComplexFloat> reference_fft_;
    std::vector<ComplexFloat> input_fft_;
    std::vector<ComplexFloat> correlation_fft_;
    std::vector<ComplexFloat> correlation_ifft_;
    std::vector<float> peaks_;

    // Метаданные
    Step current_step_;
    std::string timestamp_;
    
    // Размеры данных
    int num_shifts_;
    int num_signals_;
    size_t fft_size_;
    int num_output_points_;

    std::string getCurrentTimestamp() const {
        auto now = std::time(nullptr);
        struct tm timeinfo;
        #if defined(_WIN32) || defined(_WIN64)
            localtime_s(&timeinfo, &now);
        #else
            localtime_r(&now, &timeinfo);
        #endif
        
        std::ostringstream oss;
        oss << std::put_time(&timeinfo, "%Y-%m-%d %H:%M:%S");
        return oss.str();
    }

public:
    DataSnapshot() 
        : current_step_(Step::STEP1_REFERENCE_FFT),
          timestamp_(getCurrentTimestamp()),
          num_shifts_(0), num_signals_(0), fft_size_(0), num_output_points_(0) {}

    // Сохранение данных
    void saveReferenceFFT(const std::vector<ComplexFloat>& data, 
                         int num_shifts, size_t fft_size) override {
        reference_fft_ = data;
        num_shifts_ = num_shifts;
        fft_size_ = fft_size;
        current_step_ = Step::STEP1_REFERENCE_FFT;
        timestamp_ = getCurrentTimestamp();
    }

    void saveInputFFT(const std::vector<ComplexFloat>& data, 
                     int num_signals, size_t fft_size) override {
        input_fft_ = data;
        num_signals_ = num_signals;
        fft_size_ = fft_size;
        current_step_ = Step::STEP2_INPUT_FFT;
        timestamp_ = getCurrentTimestamp();
    }

    void saveCorrelationFFT(const std::vector<ComplexFloat>& data,
                           int num_signals, int num_shifts, size_t fft_size) override {
        correlation_fft_ = data;
        num_signals_ = num_signals;
        num_shifts_ = num_shifts;
        fft_size_ = fft_size;
        current_step_ = Step::STEP3_CORRELATION_FFT;
        timestamp_ = getCurrentTimestamp();
    }

    void saveCorrelationIFFT(const std::vector<ComplexFloat>& data,
                             int num_signals, int num_shifts, size_t fft_size) override {
        correlation_ifft_ = data;
        num_signals_ = num_signals;
        num_shifts_ = num_shifts;
        fft_size_ = fft_size;
        current_step_ = Step::STEP3_CORRELATION_IFFT;
        timestamp_ = getCurrentTimestamp();
    }

    void savePeaks(const std::vector<float>& peaks,
                  int num_signals, int num_shifts, int num_points) override {
        peaks_ = peaks;
        num_signals_ = num_signals;
        num_shifts_ = num_shifts;
        num_output_points_ = num_points;
        current_step_ = Step::STEP3_PEAKS;
        timestamp_ = getCurrentTimestamp();
    }

    // Получение данных
    const std::vector<ComplexFloat>& getReferenceFFT() const override {
        return reference_fft_;
    }

    const std::vector<ComplexFloat>& getInputFFT() const override {
        return input_fft_;
    }

    const std::vector<ComplexFloat>& getCorrelationFFT() const override {
        return correlation_fft_;
    }

    const std::vector<ComplexFloat>& getCorrelationIFFT() const override {
        return correlation_ifft_;
    }

    const std::vector<float>& getPeaks() const override {
        return peaks_;
    }

    // Метаданные
    Step getStep() const override { return current_step_; }
    std::string getTimestamp() const override { return timestamp_; }
    
    size_t getDataSize() const override {
        size_t total = 0;
        total += reference_fft_.size() * sizeof(ComplexFloat);
        total += input_fft_.size() * sizeof(ComplexFloat);
        total += correlation_fft_.size() * sizeof(ComplexFloat);
        total += correlation_ifft_.size() * sizeof(ComplexFloat);
        total += peaks_.size() * sizeof(float);
        return total;
    }

    // Экспорт в JSON
    std::string toJSON() const override {
        return toJSON(current_step_);
    }

    std::string toJSON(Step step) const override {
        std::ostringstream oss;
        oss << "{\n"
            << "  \"step\": \"" << stepToString(step) << "\",\n"
            << "  \"timestamp\": \"" << timestamp_ << "\",\n"
            << "  \"data_size_bytes\": " << getDataSize() << ",\n";

        switch (step) {
            case Step::STEP1_REFERENCE_FFT:
                oss << "  \"reference_fft\": " << complexArrayToJSON(reference_fft_) << ",\n"
                    << "  \"num_shifts\": " << num_shifts_ << ",\n"
                    << "  \"fft_size\": " << fft_size_ << "\n";
                break;
            case Step::STEP2_INPUT_FFT:
                oss << "  \"input_fft\": " << complexArrayToJSON(input_fft_) << ",\n"
                    << "  \"num_signals\": " << num_signals_ << ",\n"
                    << "  \"fft_size\": " << fft_size_ << "\n";
                break;
            case Step::STEP3_CORRELATION_FFT:
                oss << "  \"correlation_fft\": " << complexArrayToJSON(correlation_fft_) << ",\n"
                    << "  \"num_signals\": " << num_signals_ << ",\n"
                    << "  \"num_shifts\": " << num_shifts_ << ",\n"
                    << "  \"fft_size\": " << fft_size_ << "\n";
                break;
            case Step::STEP3_CORRELATION_IFFT:
                oss << "  \"correlation_ifft\": " << complexArrayToJSON(correlation_ifft_) << ",\n"
                    << "  \"num_signals\": " << num_signals_ << ",\n"
                    << "  \"num_shifts\": " << num_shifts_ << ",\n"
                    << "  \"fft_size\": " << fft_size_ << "\n";
                break;
            case Step::STEP3_PEAKS:
                oss << "  \"peaks\": " << floatArrayToJSON(peaks_) << ",\n"
                    << "  \"num_signals\": " << num_signals_ << ",\n"
                    << "  \"num_shifts\": " << num_shifts_ << ",\n"
                    << "  \"num_output_points\": " << num_output_points_ << "\n";
                break;
        }

        oss << "}";
        return oss.str();
    }

    // Статистика
    std::string getStatistics() const override {
        std::ostringstream oss;
        oss << "Data Snapshot Statistics:\n"
            << "  Timestamp: " << timestamp_ << "\n"
            << "  Current Step: " << stepToString(current_step_) << "\n"
            << "  Total Data Size: " << getDataSize() << " bytes\n";
        
        if (!reference_fft_.empty()) {
            oss << "  Reference FFT: " << reference_fft_.size() << " complex samples\n";
        }
        if (!input_fft_.empty()) {
            oss << "  Input FFT: " << input_fft_.size() << " complex samples\n";
        }
        if (!correlation_fft_.empty()) {
            oss << "  Correlation FFT: " << correlation_fft_.size() << " complex samples\n";
        }
        if (!correlation_ifft_.empty()) {
            oss << "  Correlation IFFT: " << correlation_ifft_.size() << " complex samples\n";
        }
        if (!peaks_.empty()) {
            oss << "  Peaks: " << peaks_.size() << " float values\n";
        }
        
        return oss.str();
    }

private:
    std::string stepToString(Step step) const {
        switch (step) {
            case Step::STEP1_REFERENCE_FFT: return "STEP1_REFERENCE_FFT";
            case Step::STEP2_INPUT_FFT: return "STEP2_INPUT_FFT";
            case Step::STEP3_CORRELATION_FFT: return "STEP3_CORRELATION_FFT";
            case Step::STEP3_CORRELATION_IFFT: return "STEP3_CORRELATION_IFFT";
            case Step::STEP3_PEAKS: return "STEP3_PEAKS";
            default: return "UNKNOWN";
        }
    }

    std::string complexArrayToJSON(const std::vector<ComplexFloat>& data) const {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < data.size(); ++i) {
            if (i > 0) oss << ",";
            oss << "{\"real\":" << std::fixed << std::setprecision(6) << data[i].real
                << ",\"imag\":" << data[i].imag << "}";
        }
        oss << "]";
        return oss.str();
    }

    std::string floatArrayToJSON(const std::vector<float>& data) const {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < data.size(); ++i) {
            if (i > 0) oss << ",";
            oss << std::fixed << std::setprecision(6) << data[i];
        }
        oss << "]";
        return oss.str();
    }
};

} // namespace Correlator

#endif // CORRELATOR_DATA_SNAPSHOT_HPP

