#ifndef PROFILER_HPP
#define PROFILER_HPP

#include <chrono>
#include <map>
#include <vector>
#include <string>
#include <cstdio>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <cerrno>
#include <filesystem>
#include <CL/cl.h>
#include <cmath>

/**
 * @class Profiler
 * @brief Профилирование CPU и GPU операций
 * 
 * Поддерживает:
 * - CPU таймеры (std::chrono high_resolution_clock)
 * - GPU события (OpenCL clGetEventProfilingInfo)
 * - Статистика (min, max, avg)
 * - Автоматический вывод результатов
 */
class Profiler {
public:
    enum TimeUnit {
        MICROSECONDS = 0,  // μs
        MILLISECONDS = 1,  // ms
        SECONDS = 2        // s
    };

private:
    struct TimingData {
        std::vector<double> measurements;  // в микросекундах
        std::string label;
        TimeUnit display_unit;
        
        double get_min() const {
            if (measurements.empty()) return 0.0;
            return *std::min_element(measurements.begin(), measurements.end());
        }
        
        double get_max() const {
            if (measurements.empty()) return 0.0;
            return *std::max_element(measurements.begin(), measurements.end());
        }
        
        double get_avg() const {
            if (measurements.empty()) return 0.0;
            double sum = 0.0;
            for (double m : measurements) sum += m;
            return sum / measurements.size();
        }
        
        void print(const char* format = nullptr) const {
            const char* unit_str;
            double divisor;
            
            switch (display_unit) {
                case MILLISECONDS:
                    unit_str = "ms";
                    divisor = 1000.0;
                    break;
                case SECONDS:
                    unit_str = "s";
                    divisor = 1000000.0;
                    break;
                case MICROSECONDS:
                default:
                    unit_str = "μs";
                    divisor = 1.0;
                    break;
            }
            
            if (measurements.size() == 1) {
                printf("  %-40s: %10.3f %s\n", 
                       label.c_str(), 
                       measurements[0] / divisor,
                       unit_str);
            } else {
                printf("  %-40s: avg=%-10.3f min=%-10.3f max=%-10.3f %s (n=%zu)\n",
                       label.c_str(),
                       get_avg() / divisor,
                       get_min() / divisor,
                       get_max() / divisor,
                       unit_str,
                       measurements.size());
            }
        }
    };
    
    std::map<std::string, TimingData> timings;
    std::map<std::string, std::chrono::high_resolution_clock::time_point> start_times;

public:
    Profiler() = default;
    ~Profiler() = default;
    
    /**
     * Начать отсчёт времени для CPU операции
     * @param label уникальный идентификатор операции
     */
    void start(const std::string& label) {
        start_times[label] = std::chrono::high_resolution_clock::now();
    }
    
    /**
     * Завершить отсчёт и сохранить результат
     * @param label уникальный идентификатор операции
     * @param unit единица измерения для вывода
     * @return время в микросекундах
     */
    double stop(const std::string& label, TimeUnit unit = MICROSECONDS) {
        auto end = std::chrono::high_resolution_clock::now();
        auto it = start_times.find(label);
        
        if (it == start_times.end()) {
            fprintf(stderr, "ERROR: No start time found for label '%s'\n", label.c_str());
            return 0.0;
        }
        
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
            end - it->second
        ).count();
        
        // Инициализировать если новый лейбл
        if (timings.find(label) == timings.end()) {
            timings[label] = TimingData{
                .measurements = std::vector<double>(),
                .label = label,
                .display_unit = unit
            };
        }
        
        timings[label].measurements.push_back(static_cast<double>(duration_us));
        start_times.erase(it);
        
        return static_cast<double>(duration_us);
    }
    
    /**
     * Профилировать OpenCL событие
     * @param event OpenCL события (должен быть создан с CL_PROFILING_ENABLE)
     * @param label уникальный идентификатор операции
     * @param unit единица измерения для вывода
     * @return время выполнения в микросекундах
     */
    double profile_cl_event(cl_event event, const std::string& label, TimeUnit unit = MICROSECONDS) {
        cl_int err;
        cl_ulong time_start, time_end;
        
        // Дождаться завершения события
        err = clWaitForEvents(1, &event);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: clWaitForEvents failed with code %d\n", err);
            return 0.0;
        }
        
        // Получить временные метки
        err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, 
                                      sizeof(time_start), &time_start, nullptr);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: clGetEventProfilingInfo START failed with code %d\n", err);
            return 0.0;
        }
        
        err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, 
                                      sizeof(time_end), &time_end, nullptr);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: clGetEventProfilingInfo END failed with code %d\n", err);
            return 0.0;
        }
        
        // Время в наносекундах → микросекундах
        double duration_us = (time_end - time_start) / 1000.0;
        
        // Инициализировать если новый лейбл
        if (timings.find(label) == timings.end()) {
            timings[label] = TimingData{
                .measurements = std::vector<double>(),
                .label = label,
                .display_unit = unit
            };
        }
        
        timings[label].measurements.push_back(duration_us);
        
        return duration_us;
    }
    
    /**
     * Получить последнее измерение в микросекундах
     */
    double get_last(const std::string& label) const {
        auto it = timings.find(label);
        if (it == timings.end() || it->second.measurements.empty()) {
            return 0.0;
        }
        return it->second.measurements.back();
    }
    
    /**
     * Получить среднее значение в микросекундах
     */
    double get_avg(const std::string& label) const {
        auto it = timings.find(label);
        if (it == timings.end()) {
            return 0.0;
        }
        return it->second.get_avg();
    }
    
    /**
     * Получить минимальное значение в микросекундах
     */
    double get_min(const std::string& label) const {
        auto it = timings.find(label);
        if (it == timings.end()) {
            return 0.0;
        }
        return it->second.get_min();
    }
    
    /**
     * Получить максимальное значение в микросекундах
     */
    double get_max(const std::string& label) const {
        auto it = timings.find(label);
        if (it == timings.end()) {
            return 0.0;
        }
        return it->second.get_max();
    }
    
    /**
     * Получить количество измерений
     */
    size_t get_count(const std::string& label) const {
        auto it = timings.find(label);
        if (it == timings.end()) {
            return 0;
        }
        return it->second.measurements.size();
    }
    
    /**
     * Сумма всех измерений в микросекундах
     */
    double get_total(const std::string& label) const {
        auto it = timings.find(label);
        if (it == timings.end()) {
            return 0.0;
        }
        double sum = 0.0;
        for (double m : it->second.measurements) {
            sum += m;
        }
        return sum;
    }
    
    /**
     * Сумма всех измерений для всех меток в микросекундах
     */
    double get_total_all() const {
        double sum = 0.0;
        for (const auto& [label, data] : timings) {
            for (double m : data.measurements) {
                sum += m;
            }
        }
        return sum;
    }
    
    /**
     * Вывести одно измерение
     */
    void print(const std::string& label) const {
        auto it = timings.find(label);
        if (it != timings.end()) {
            it->second.print();
        } else {
            printf("  %-40s: NOT FOUND\n", label.c_str());
        }
    }
    
    /**
     * Вывести все измерения с заголовком
     */
    void print_all(const std::string& title = "PROFILING RESULTS") const {
        printf("\n");
        printf("====== %s ======\n", title.c_str());
        for (const auto& [label, data] : timings) {
            data.print();
        }
        printf("======== TOTAL TIME (all ops): %.3f ms ========\n\n", 
               get_total_all() / 1000.0);
    }
    
    /**
     * Вывести сравнение двух вариантов
     */
    void compare_variants(
        const std::string& variant1_name,
        const std::vector<std::string>& variant1_labels,
        const std::string& variant2_name,
        const std::vector<std::string>& variant2_labels
    ) const {
        double total1 = 0.0, total2 = 0.0;
        
        printf("\n");
        printf("========== VARIANT COMPARISON ==========\n");
        printf("\n%s:\n", variant1_name.c_str());
        for (const auto& label : variant1_labels) {
            auto it = timings.find(label);
            if (it != timings.end()) {
                double avg_ms = it->second.get_avg() / 1000.0;
                printf("  %-40s: %.3f ms\n", label.c_str(), avg_ms);
                total1 += avg_ms;
            }
        }
        printf("  %-40s: %.3f ms\n", "TOTAL", total1);
        
        printf("\n%s:\n", variant2_name.c_str());
        for (const auto& label : variant2_labels) {
            auto it = timings.find(label);
            if (it != timings.end()) {
                double avg_ms = it->second.get_avg() / 1000.0;
                printf("  %-40s: %.3f ms\n", label.c_str(), avg_ms);
                total2 += avg_ms;
            }
        }
        printf("  %-40s: %.3f ms\n", "TOTAL", total2);
        
        printf("\n");
        if (total1 < total2) {
            double gain = (total2 - total1) / total2 * 100.0;
            printf("🏆 WINNER: %s (%.1f%% faster)\n", variant1_name.c_str(), gain);
        } else if (total2 < total1) {
            double gain = (total1 - total2) / total1 * 100.0;
            printf("🏆 WINNER: %s (%.1f%% faster)\n", variant2_name.c_str(), gain);
        } else {
            printf("⚖️  EQUAL: Both variants take the same time\n");
        }
        printf("=========================================\n\n");
    }
    
    /**
     * Очистить все замеры
     */
    void clear() {
        timings.clear();
        start_times.clear();
    }
    
    /**
     * Очистить замеры для конкретной метки
     */
    void clear(const std::string& label) {
        auto it = timings.find(label);
        if (it != timings.end()) {
            it->second.measurements.clear();
        }
    }
    
    /**
     * Структура с информацией о GPU
     */
    struct GPUInfo {
        std::string device_name;
        std::string driver_version;
        std::string api_version;
    };
    
    /**
     * Получить информацию о GPU через OpenCL
     */
    static GPUInfo get_gpu_info(cl_device_id device_id) {
        GPUInfo info;
        
        char device_name[1024] = {0};
        char driver_version[256] = {0};
        char device_version[256] = {0};
        
        clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
        clGetDeviceInfo(device_id, CL_DRIVER_VERSION, sizeof(driver_version), driver_version, nullptr);
        clGetDeviceInfo(device_id, CL_DEVICE_VERSION, sizeof(device_version), device_version, nullptr);
        
        info.device_name = device_name ? device_name : "Unknown";
        info.driver_version = driver_version ? driver_version : "Unknown";
        info.api_version = device_version ? device_version : "Unknown";
        
        return info;
    }
    
    /**
     * Экспортировать профилирование в Markdown файл
     * @param filename путь к файлу для сохранения (будет переименован с timestamp)
     * @param step_details дополнительные детали по шагам (Step1, Step2, Step3)
     * @param gpu_info информация о GPU
     */
    bool export_to_markdown(
        const std::string& base_filename,
        const std::map<std::string, std::map<std::string, double>>& step_details = {},
        const GPUInfo& gpu_info = {"Unknown", "Unknown", "Unknown"}
    ) const {
        // Создать директорию, если она не существует
        try {
            std::filesystem::path base_path(base_filename);
            if (base_path.has_parent_path()) {
                std::filesystem::create_directories(base_path.parent_path());
            }
        } catch (const std::exception& e) {
            fprintf(stderr, "WARNING: Cannot create directory for report: %s\n", e.what());
        }
        
        // Получить текущую дату и время (безопасный способ для VS2022)
        auto now = std::time(nullptr);
        struct tm timeinfo;
        #if defined(_WIN32) || defined(_WIN64)
            errno_t err = localtime_s(&timeinfo, &now);
            if (err != 0) {
                // Если ошибка, используем текущее время как fallback
                timeinfo = {};
            }
        #else
            localtime_r(&now, &timeinfo);
        #endif
        
        // Форматировать дату для имени файла: YYYY-MM-DD_HH-MM-SS
        char timestamp_str[100];
        std::strftime(timestamp_str, sizeof(timestamp_str), "%Y-%m-%d_%H-%M-%S", &timeinfo);
        
        // Форматировать дату для отчета: YYYY-MM-DD HH:MM:SS
        char datetime_str[100];
        std::strftime(datetime_str, sizeof(datetime_str), "%Y-%m-%d %H:%M:%S", &timeinfo);
        
        // Создать имя файла с timestamp
        std::filesystem::path base_path(base_filename);
        std::filesystem::path dir = base_path.parent_path();
        
        // Если parent_path пустой (например, просто "profiling.md"), используем текущую директорию
        if (dir.empty() || dir.string() == ".") {
            dir = std::filesystem::current_path() / "Report";
            std::filesystem::create_directories(dir);
        }
        
        std::string stem = base_path.stem().string();
        std::string ext = base_path.extension().string();
        
        // Формат: profiling_2025-12-21_16-51-59.md
        std::string filename_with_timestamp = (dir / (stem + "_" + std::string(timestamp_str) + ext)).string();
        
        // Отладочный вывод
        fprintf(stdout, "[DEBUG EXPORT] Creating report: %s\n", filename_with_timestamp.c_str());
        fprintf(stdout, "[DEBUG EXPORT] Directory: %s (exists: %s)\n", dir.string().c_str(), 
                std::filesystem::exists(dir) ? "yes" : "no");
        fprintf(stdout, "[DEBUG EXPORT] Number of timings in profiler: %zu\n", timings.size());
        for (const auto& [label, data] : timings) {
            fprintf(stdout, "[DEBUG EXPORT]   Timing: %s -> %zu measurements\n", label.c_str(), data.measurements.size());
        }
        
        std::ofstream file(filename_with_timestamp);
        if (!file.is_open()) {
            fprintf(stderr, "ERROR: Cannot open file for writing: %s\n", filename_with_timestamp.c_str());
            fprintf(stderr, "ERROR: Directory exists: %s\n", std::filesystem::exists(dir) ? "yes" : "no");
            return false;
        }
        
        // Заголовок отчета
        file << "# 📊 Отчет профилирования FFT Correlator\n\n";
        file << "**Дата создания:** " << datetime_str << "\n\n";
        file << "**Режим сборки:** Release\n\n";
        file << "---\n\n";
        
        // Информация о GPU
        file << "## 🖥️ Информация о системе\n\n";
        file << "| Параметр | Значение |\n";
        file << "|----------|----------|\n";
        file << "| **GPU** | " << gpu_info.device_name << " |\n";
        file << "| **Драйвер** | " << gpu_info.driver_version << " |\n";
        file << "| **API версия** | " << gpu_info.api_version << " |\n";
        file << "| **Timestamp** | " << timestamp_str << " |\n";
        file << "\n";
        file << "**Примечание:** GPU времена измеряются от момента постановки в очередь (QUEUED) до завершения выполнения (END)\n";
        file << "\n";
        file << "---\n\n";
        
        // Общая статистика
        file << "## 📈 Общая статистика\n\n";
        file << "| Метрика | Значение |\n";
        file << "|---------|----------|\n";
        file << "| Общее время выполнения | " << std::fixed << std::setprecision(3) 
             << get_total_all() / 1000.0 << " ms |\n";
        file << "| Количество профилированных операций | " << timings.size() << " |\n";
        file << "\n";
        
        // Профилирование по шагам
        file << "## 🔄 Профилирование по шагам\n\n";
        
        // Step 1
        if (timings.find("Step1_Total") != timings.end()) {
            file << "### Step 1: Обработка опорных сигналов\n\n";
            double step1_total_ms = get_avg("Step1_Total") / 1000.0;
            
            // Вычислить общее время на GPU
            double step1_gpu_total = 0.0;
            if (step_details.find("Step1") != step_details.end()) {
                for (const auto& [op, time_ms] : step_details.at("Step1")) {
                    if (op.find("total GPU time") != std::string::npos) {
                        step1_gpu_total += time_ms;
                    }
                }
            }
            
            file << "**Общее время на GPU:** " << std::fixed << std::setprecision(3) 
                 << step1_gpu_total << " ms\n";
            file << "**Общее время Step 1:** " << std::fixed << std::setprecision(3) 
                 << step1_total_ms << " ms\n\n";
            
            file << "*Примечание: Pre-callback (int32 → float2 конвертация) встроен в FFT план через clfftSetPlanCallback и выполняется автоматически. Время callback включено в время FFT операции.*\n\n";
            
            if (step_details.find("Step1") != step_details.end() && !step_details.at("Step1").empty()) {
                file << "| Операция | Время (ms) |\n";
                file << "|----------|------------|\n";
                double step1_sum = 0.0;
                double step1_gpu_sum = 0.0;
                for (const auto& [op, time_ms] : step_details.at("Step1")) {
                    file << "| " << op << " | " << std::fixed << std::setprecision(3) 
                         << time_ms << " |\n";
                    step1_sum += time_ms;
                    // Суммируем только GPU времена (total GPU time)
                    if (op.find("total GPU time") != std::string::npos) {
                        step1_gpu_sum += time_ms;
                    }
                }
                // Вычислить недостающее время (overhead между операциями)
                double step1_overhead = step1_total_ms - step1_sum;
                if (step1_overhead > 0.001) {  // Добавляем только если есть заметная разница
                    file << "| **Overhead** | " << std::fixed << std::setprecision(3) 
                         << step1_overhead << " |\n";
                    file << "| *Overhead включает: printf, подготовка параметров, время между операциями, вызов функций* |\n";
                }
                file << "| **ИТОГО GPU** | **" << std::fixed << std::setprecision(3) 
                     << step1_gpu_sum << "** |\n";
                file << "| **ИТОГО** | **" << std::fixed << std::setprecision(3) 
                     << step1_total_ms << "** |\n\n";
            } else {
                file << "*Детальные данные для Step 1 отсутствуют*\n\n";
            }
        } else {
            file << "### Step 1: Обработка опорных сигналов\n\n";
            file << "*Данные для Step 1 не найдены*\n\n";
        }
        
        // Step 2
        if (timings.find("Step2_Total") != timings.end()) {
            file << "### Step 2: Обработка входных сигналов\n\n";
            double step2_total_ms = get_avg("Step2_Total") / 1000.0;
            
            // Вычислить общее время на GPU
            double step2_gpu_total = 0.0;
            if (step_details.find("Step2") != step_details.end()) {
                for (const auto& [op, time_ms] : step_details.at("Step2")) {
                    if (op.find("total GPU time") != std::string::npos) {
                        step2_gpu_total += time_ms;
                    }
                }
            }
            
            file << "**Общее время на GPU:** " << std::fixed << std::setprecision(3) 
                 << step2_gpu_total << " ms\n";
            file << "**Общее время Step 2:** " << std::fixed << std::setprecision(3) 
                 << step2_total_ms << " ms\n\n";
            
            file << "*Примечание: Pre-callback (int32 → float2 конвертация) встроен в FFT план через clfftSetPlanCallback и выполняется автоматически. Время callback включено в время FFT операции.*\n\n";
            
            if (step_details.find("Step2") != step_details.end() && !step_details.at("Step2").empty()) {
                file << "| Операция | Время (ms) |\n";
                file << "|----------|------------|\n";
                double step2_sum = 0.0;
                double step2_gpu_sum = 0.0;
                for (const auto& [op, time_ms] : step_details.at("Step2")) {
                    file << "| " << op << " | " << std::fixed << std::setprecision(3) 
                         << time_ms << " |\n";
                    step2_sum += time_ms;
                    // Суммируем только GPU времена (total GPU time)
                    if (op.find("total GPU time") != std::string::npos) {
                        step2_gpu_sum += time_ms;
                    }
                }
                // Вычислить недостающее время (overhead между операциями)
                double step2_overhead = step2_total_ms - step2_sum;
                if (step2_overhead > 0.001) {  // Добавляем только если есть заметная разница
                    file << "| **Overhead** | " << std::fixed << std::setprecision(3) 
                         << step2_overhead << " |\n";
                    file << "| *Overhead включает: printf, подготовка параметров, время между операциями, вызов функций* |\n";
                }
                file << "| **ИТОГО GPU** | **" << std::fixed << std::setprecision(3) 
                     << step2_gpu_sum << "** |\n";
                file << "| **ИТОГО** | **" << std::fixed << std::setprecision(3) 
                     << step2_total_ms << "** |\n\n";
            } else {
                file << "*Детальные данные для Step 2 отсутствуют*\n\n";
            }
        } else {
            file << "### Step 2: Обработка входных сигналов\n\n";
            file << "*Данные для Step 2 не найдены*\n\n";
        }
        
        // Step 3
        if (timings.find("Step3_Total") != timings.end()) {
            file << "### Step 3: Корреляция\n\n";
            double step3_total_ms = get_avg("Step3_Total") / 1000.0;
            
            // Вычислить общее время на GPU
            double step3_gpu_total = 0.0;
            if (step_details.find("Step3") != step_details.end()) {
                for (const auto& [op, time_ms] : step_details.at("Step3")) {
                    if (op.find("total GPU time") != std::string::npos) {
                        step3_gpu_total += time_ms;
                    }
                }
            }
            
            file << "**Общее время на GPU:** " << std::fixed << std::setprecision(3) 
                 << step3_gpu_total << " ms\n";
            file << "**Общее время Step 3:** " << std::fixed << std::setprecision(3) 
                 << step3_total_ms << " ms\n\n";
            
            file << "*Примечания:\n";
            file << "- Pre-callback (Complex Multiply - перемножение спектров) ВСТРОЕН в IFFT план через clfftSetPlanCallback и выполняется автоматически. Время callback включено в время IFFT операции.\n";
            file << "- Post-callback (find peaks) встроен в IFFT план через clfftSetPlanCallback и выполняется автоматически. Время callback включено в время IFFT операции.\n";
            file << "- Оба callback'а выполняются БЕЗ дополнительных синхронизаций, что обеспечивает минимальное время работы.*\n\n";
            
            if (step_details.find("Step3") != step_details.end() && !step_details.at("Step3").empty()) {
                file << "| Операция | Время (ms) |\n";
                file << "|----------|------------|\n";
                double step3_sum = 0.0;
                double step3_gpu_sum = 0.0;
                for (const auto& [op, time_ms] : step_details.at("Step3")) {
                    file << "| " << op << " | " << std::fixed << std::setprecision(3) 
                         << time_ms << " |\n";
                    step3_sum += time_ms;
                    // Суммируем только GPU времена (total GPU time)
                    if (op.find("total GPU time") != std::string::npos) {
                        step3_gpu_sum += time_ms;
                    }
                }
                // Вычислить недостающее время (overhead между операциями)
                double step3_overhead = step3_total_ms - step3_sum;
                if (step3_overhead > 0.001) {  // Добавляем только если есть заметная разница
                    file << "| **Overhead** | " << std::fixed << std::setprecision(3) 
                         << step3_overhead << " |\n";
                    file << "| *Overhead включает: printf, подготовка параметров, время между операциями, вызов функций* |\n";
                }
                file << "| **ИТОГО GPU** | **" << std::fixed << std::setprecision(3) 
                     << step3_gpu_sum << "** |\n";
                file << "| **ИТОГО** | **" << std::fixed << std::setprecision(3) 
                     << step3_total_ms << "** |\n\n";
            } else {
                file << "*Детальные данные для Step 3 отсутствуют*\n\n";
            }
        } else {
            file << "### Step 3: Корреляция\n\n";
            file << "*Данные для Step 3 не найдены*\n\n";
        }
        
        // Детальное профилирование по времени (только GPU времена)
        file << "## ⏱️ Детальное профилирование по времени\n\n";
        
        // Собрать все GPU операции из step_details
        std::vector<std::pair<std::string, double>> gpu_operations;
        
        // Step 1 GPU операции
        if (step_details.find("Step1") != step_details.end()) {
            for (const auto& [op, time_ms] : step_details.at("Step1")) {
                if (op.find("total GPU time") != std::string::npos) {
                    gpu_operations.push_back({"Step 1: " + op, time_ms});
        }
            }
        }
        
        // Step 2 GPU операции
        if (step_details.find("Step2") != step_details.end()) {
            for (const auto& [op, time_ms] : step_details.at("Step2")) {
                if (op.find("total GPU time") != std::string::npos) {
                    gpu_operations.push_back({"Step 2: " + op, time_ms});
                }
            }
        }
        
        // Step 3 GPU операции
        if (step_details.find("Step3") != step_details.end()) {
            for (const auto& [op, time_ms] : step_details.at("Step3")) {
                if (op.find("total GPU time") != std::string::npos) {
                    gpu_operations.push_back({"Step 3: " + op, time_ms});
                }
            }
        }
        
        if (gpu_operations.empty()) {
            file << "*Нет данных профилирования GPU*\n\n";
        } else {
            file << "| Операция | Время GPU (ms) |\n";
            file << "|----------|-----------------|\n";
            
            for (const auto& [op_name, time_ms] : gpu_operations) {
                file << "| " << op_name << " | " << std::fixed << std::setprecision(3) << time_ms << " |\n";
            }
            
            // Добавить общее время на GPU для каждого шага
            file << "\n";
            file << "| **Общее время на GPU** | **Время (ms)** |\n";
            file << "|------------------------|-----------------|\n";
            
            double step1_gpu_total = 0.0;
            double step2_gpu_total = 0.0;
            double step3_gpu_total = 0.0;
            
            if (step_details.find("Step1") != step_details.end()) {
                for (const auto& [op, time_ms] : step_details.at("Step1")) {
                    if (op.find("total GPU time") != std::string::npos) {
                        step1_gpu_total += time_ms;
                    }
                }
                file << "| **Step 1** | **" << std::fixed << std::setprecision(3) << step1_gpu_total << "** |\n";
            }
            
            if (step_details.find("Step2") != step_details.end()) {
                for (const auto& [op, time_ms] : step_details.at("Step2")) {
                    if (op.find("total GPU time") != std::string::npos) {
                        step2_gpu_total += time_ms;
                    }
                }
                file << "| **Step 2** | **" << std::fixed << std::setprecision(3) << step2_gpu_total << "** |\n";
            }
            
            if (step_details.find("Step3") != step_details.end()) {
                for (const auto& [op, time_ms] : step_details.at("Step3")) {
                    if (op.find("total GPU time") != std::string::npos) {
                        step3_gpu_total += time_ms;
                    }
                }
                file << "| **Step 3** | **" << std::fixed << std::setprecision(3) << step3_gpu_total << "** |\n";
            }
            
            // Добавить суммарное время
            double total_gpu_time = step1_gpu_total + step2_gpu_total + step3_gpu_total;
            file << "| **ВСЕГО** | **" << std::fixed << std::setprecision(3) << total_gpu_time << "** |\n";
            
            file << "\n";
        }
        
        // Футер
        file << "---\n\n";
        file << "*Отчет сгенерирован автоматически системой профилирования*\n";
        
        file.flush(); // Ensure data is written to disk
        file.close();
        
        // Проверить, что файл создан
        if (!std::filesystem::exists(filename_with_timestamp)) {
            fprintf(stderr, "ERROR: Report file was not created: %s\n", filename_with_timestamp.c_str());
            return false;
        }
        
        auto abs_path = std::filesystem::absolute(filename_with_timestamp);
        auto file_size = std::filesystem::file_size(filename_with_timestamp);
        
        fprintf(stdout, "[SUCCESS] Отчет создан успешно!\n");
        fprintf(stdout, "[SUCCESS] Имя файла: %s\n", std::filesystem::path(filename_with_timestamp).filename().string().c_str());
        fprintf(stdout, "[SUCCESS] Полный путь: %s\n", abs_path.string().c_str());
        fprintf(stdout, "[SUCCESS] Размер файла: %lld bytes\n", static_cast<long long>(file_size));
        
        if (file_size == 0) {
            fprintf(stderr, "[WARNING] Файл отчета пустой (0 bytes)!\n");
            return false;
        }
        
        return true;
    }
    
    /**
     * Экспортировать профилирование в JSON файл
     * @param base_filename путь к файлу для сохранения (будет переименован с timestamp)
     * @param step_details дополнительные детали по шагам (Step1, Step2, Step3)
     * @param gpu_info информация о GPU
     */
    bool export_to_json(
        const std::string& base_filename,
        const std::map<std::string, std::map<std::string, double>>& step_details = {},
        const GPUInfo& gpu_info = {"Unknown", "Unknown", "Unknown"}
    ) const {
        // Создать директорию JSON, если она не существует
        try {
            std::filesystem::path base_path(base_filename);
            std::filesystem::path json_dir = base_path.parent_path() / "JSON";
            std::filesystem::create_directories(json_dir);
        } catch (const std::exception& e) {
            fprintf(stderr, "WARNING: Cannot create JSON directory: %s\n", e.what());
        }
        
        // Получить текущую дату и время (используем тот же timestamp что и для MD)
        auto now = std::time(nullptr);
        struct tm timeinfo;
        #if defined(_WIN32) || defined(_WIN64)
            errno_t err = localtime_s(&timeinfo, &now);
            if (err != 0) {
                timeinfo = {};
            }
        #else
            localtime_r(&now, &timeinfo);
        #endif
        
        // Форматировать дату для имени файла: YYYY-MM-DD_HH-MM-SS
        char timestamp_str[100];
        std::strftime(timestamp_str, sizeof(timestamp_str), "%Y-%m-%d_%H-%M-%S", &timeinfo);
        
        // Форматировать дату для отчета: YYYY-MM-DD HH:MM:SS
        char datetime_str[100];
        std::strftime(datetime_str, sizeof(datetime_str), "%Y-%m-%d %H:%M:%S", &timeinfo);
        
        // Создать имя файла с timestamp в директории JSON
        std::filesystem::path base_path(base_filename);
        std::filesystem::path json_dir = base_path.parent_path() / "JSON";
        std::string stem = base_path.stem().string();
        
        // Формат: profiling_2025-12-21_16-51-59.json
        std::string json_filename = (json_dir / (stem + "_" + std::string(timestamp_str) + ".json")).string();
        
        std::ofstream file(json_filename);
        if (!file.is_open()) {
            fprintf(stderr, "ERROR: Cannot open JSON file for writing: %s\n", json_filename.c_str());
            return false;
        }
        
        // Вспомогательная функция для экранирования JSON строк
        auto escape_json = [](const std::string& str) -> std::string {
            std::string escaped;
            for (char c : str) {
                switch (c) {
                    case '"': escaped += "\\\""; break;
                    case '\\': escaped += "\\\\"; break;
                    case '\n': escaped += "\\n"; break;
                    case '\r': escaped += "\\r"; break;
                    case '\t': escaped += "\\t"; break;
                    default: escaped += c; break;
                }
            }
            return escaped;
        };
        
        // Форматирование числа для JSON
        auto format_double = [](double value) -> std::string {
            if (std::isnan(value) || std::isinf(value)) {
                return "null";
            }
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << value;
            return oss.str();
        };
        
        file << "{\n";
        
        // Шапка - информация о системе
        file << "  \"report_info\": {\n";
        file << "    \"title\": \"Отчет профилирования FFT Correlator\",\n";
        file << "    \"creation_date\": \"" << datetime_str << "\",\n";
        file << "    \"build_mode\": \"Release\",\n";
        file << "    \"timestamp\": \"" << timestamp_str << "\"\n";
        file << "  },\n";
        
        // Информация о системе (из шапки MD)
        file << "  \"system_info\": {\n";
        file << "    \"gpu\": \"" << escape_json(gpu_info.device_name) << "\",\n";
        file << "    \"driver_version\": \"" << escape_json(gpu_info.driver_version) << "\",\n";
        file << "    \"api_version\": \"" << escape_json(gpu_info.api_version) << "\",\n";
        file << "    \"timestamp\": \"" << timestamp_str << "\",\n";
        file << "    \"note\": \"GPU времена измеряются от момента постановки в очередь (QUEUED) до завершения выполнения (END)\"\n";
        file << "  },\n";
        
        // Общая статистика
        file << "  \"summary\": {\n";
        file << "    \"total_execution_time_ms\": " << format_double(get_total_all() / 1000.0) << ",\n";
        file << "    \"profiled_operations_count\": " << timings.size() << "\n";
        file << "  },\n";
        
        // Профилирование по шагам
        file << "  \"steps\": {\n";
        
        // Step 1
        if (timings.find("Step1_Total") != timings.end()) {
            double step1_total_ms = get_avg("Step1_Total") / 1000.0;
            file << "    \"Step1\": {\n";
            file << "      \"description\": \"Обработка опорных сигналов\",\n";
            file << "      \"total_time_ms\": " << format_double(step1_total_ms) << ",\n";
            file << "      \"operations\": {\n";
            
            if (step_details.find("Step1") != step_details.end() && !step_details.at("Step1").empty()) {
                double step1_sum = 0.0;
                size_t op_count = 0;
                for (const auto& [op, time_ms] : step_details.at("Step1")) {
                    step1_sum += time_ms;
                }
                size_t total_ops = step_details.at("Step1").size();
                
                // Добавляем все операции
                for (const auto& [op, time_ms] : step_details.at("Step1")) {
                    file << "        \"" << escape_json(op) << "\": " << format_double(time_ms);
                    if (++op_count < total_ops || (step1_total_ms - step1_sum > 0.001)) file << ",";
                    file << "\n";
                }
                
                // Добавляем overhead если есть
                double step1_overhead = step1_total_ms - step1_sum;
                if (step1_overhead > 0.001) {
                    file << "        \"Other operations (overhead)\": " << format_double(step1_overhead) << "\n";
                }
            }
            
            file << "      }\n";
            file << "    }";
        }
        
        // Step 2
        bool need_comma = (timings.find("Step1_Total") != timings.end());
        if (timings.find("Step2_Total") != timings.end()) {
            if (need_comma) file << ",\n";
            double step2_total_ms = get_avg("Step2_Total") / 1000.0;
            file << "    \"Step2\": {\n";
            file << "      \"description\": \"Обработка входных сигналов\",\n";
            file << "      \"total_time_ms\": " << format_double(step2_total_ms) << ",\n";
            file << "      \"operations\": {\n";
            
            if (step_details.find("Step2") != step_details.end() && !step_details.at("Step2").empty()) {
                double step2_sum = 0.0;
                size_t op_count = 0;
                for (const auto& [op, time_ms] : step_details.at("Step2")) {
                    step2_sum += time_ms;
                }
                size_t total_ops = step_details.at("Step2").size();
                
                // Добавляем все операции
                for (const auto& [op, time_ms] : step_details.at("Step2")) {
                    file << "        \"" << escape_json(op) << "\": " << format_double(time_ms);
                    if (++op_count < total_ops || (step2_total_ms - step2_sum > 0.001)) file << ",";
                    file << "\n";
                }
                
                // Добавляем overhead если есть
                double step2_overhead = step2_total_ms - step2_sum;
                if (step2_overhead > 0.001) {
                    file << "        \"Other operations (overhead)\": " << format_double(step2_overhead) << "\n";
                }
            }
            
            file << "      }\n";
            file << "    }";
            need_comma = true;
        }
        
        // Step 3
        if (timings.find("Step3_Total") != timings.end()) {
            if (need_comma) file << ",\n";
            double step3_total_ms = get_avg("Step3_Total") / 1000.0;
            file << "    \"Step3\": {\n";
            file << "      \"description\": \"Корреляция\",\n";
            file << "      \"total_time_ms\": " << format_double(step3_total_ms) << ",\n";
            file << "      \"operations\": {\n";
            
            if (step_details.find("Step3") != step_details.end() && !step_details.at("Step3").empty()) {
                double step3_sum = 0.0;
                size_t op_count = 0;
                for (const auto& [op, time_ms] : step_details.at("Step3")) {
                    step3_sum += time_ms;
                }
                size_t total_ops = step_details.at("Step3").size();
                
                // Добавляем все операции
                for (const auto& [op, time_ms] : step_details.at("Step3")) {
                    file << "        \"" << escape_json(op) << "\": " << format_double(time_ms);
                    if (++op_count < total_ops || (step3_total_ms - step3_sum > 0.001)) file << ",";
                    file << "\n";
                }
                
                // Добавляем overhead если есть
                double step3_overhead = step3_total_ms - step3_sum;
                if (step3_overhead > 0.001) {
                    file << "        \"Other operations (overhead)\": " << format_double(step3_overhead) << "\n";
                }
            }
            
            file << "      }\n";
            file << "    }";
        }
        
        file << "\n  }\n";
        file << "}\n";
        
        file.close();
        return true;
    }
};

#endif // PROFILER_HPP
