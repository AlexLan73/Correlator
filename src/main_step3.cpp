#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cstring>
#include <CL/opencl.h>
#include <clFFT.h>
#include "profiler.hpp"
#include "fft_handler.hpp"

// ============================================================================
// КОНФИГУРАЦИЯ ШАГ 3
// ============================================================================

const size_t N = (1 << 15);           // 2^15 = 32,768
const int NUM_SHIFTS = 40;            // циклические сдвиги
const int NUM_SIGNALS = 50;           // входные сигналы
const int N_KG = 5;                   // выводимых точек
const float SCALE_FACTOR = 1.0f / 32768.0f;

// M-Sequence Generator
std::vector<int32_t> generate_m_sequence(size_t length, uint32_t seed = 0x1) {
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

// ============================================================================
// ШАГ 3: ПОЛНАЯ КОРРЕЛЯЦИЯ
// ============================================================================

void run_step3(Profiler& profiler) {
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║         FFT CORRELATOR - STEP 3: CORRELATION                 ║\n");
    printf("║     Multiply + IFFT + Post-callback (Find Peaks)             ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    cl_int err = CL_SUCCESS;
    
    // ========================================================================
    // 1. ИНИЦИАЛИЗИРОВАТЬ GPU КОНТЕКСТ
    // ========================================================================
    
    printf("[GPU] Initializing OpenCL context...\n");
    
    cl_platform_id platform_id = nullptr;
    cl_device_id device_id = nullptr;
    
    err = clGetPlatformIDs(1, &platform_id, nullptr);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: clGetPlatformIDs failed\n");
        return;
    }
    
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, nullptr);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, nullptr);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: No compute device found!\n");
            return;
        }
    }
    
    cl_context context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: clCreateContext failed\n");
        return;
    }
    
    cl_command_queue queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: clCreateCommandQueue failed\n");
        clReleaseContext(context);
        return;
    }
    
    printf("[OK] GPU context initialized\n\n");
    
    // ========================================================================
    // 2. ИНИЦИАЛИЗИРОВАТЬ clFFT
    // ========================================================================
    
    printf("[GPU] Initializing clFFT library...\n");
    
    clfftSetupData fftSetup;
    clfftInitSetupData(&fftSetup);
    clfftSetup(&fftSetup);
    
    printf("[OK] clFFT initialized\n\n");
    
    // ========================================================================
    // 3. СОЗДАТЬ FFT HANDLER
    // ========================================================================
    
    printf("[GPU] Creating FFT handler...\n");
    
    FFTHandler fft_handler(context, queue, device_id);
    
    try {
        fft_handler.initialize(N, NUM_SHIFTS, NUM_SIGNALS, N_KG, SCALE_FACTOR);
        printf("[OK] FFT handler initialized\n\n");
    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR: FFT handler initialization failed: %s\n", e.what());
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return;
    }
    
    // ========================================================================
    // 4. STEP 1: ОПОРНЫЕ СИГНАЛЫ
    // ========================================================================
    
    printf("========== STEP 1 (REFERENCE) ==========\n\n");
    
    std::vector<int32_t> reference_signal(N);
    printf("[INIT] Generating reference signal (M-sequence)...\n");
    reference_signal = generate_m_sequence(N, 0x12345678);
    printf("[OK] Reference signal ready\n\n");
    
    double time_upload_ref, time_callback_ref, time_fft_ref;
    profiler.start("Step1_Total");
    
    try {
        fft_handler.step1_reference_signals(reference_signal.data(), N, NUM_SHIFTS, SCALE_FACTOR, time_upload_ref, time_callback_ref, time_fft_ref);
    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR: Step 1 failed: %s\n", e.what());
        profiler.stop("Step1_Total", Profiler::MILLISECONDS);
        fft_handler.cleanup();
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return;
    }
    
    profiler.stop("Step1_Total", Profiler::MILLISECONDS);
    
    printf("Step 1 Results:\n");
    printf("  Upload:     %.3f ms\n", time_upload_ref);
    printf("  Callback:   %.3f ms\n", time_callback_ref);
    printf("  FFT (40):   %.3f ms\n", time_fft_ref);
    printf("  TOTAL:      %.3f ms\n\n", time_upload_ref + time_callback_ref + time_fft_ref);
    
    // ========================================================================
    // 5. STEP 2: ВХОДНЫЕ СИГНАЛЫ
    // ========================================================================
    
    printf("========== STEP 2 (INPUT SIGNALS) ==========\n\n");
    
    std::vector<int32_t> input_signals(NUM_SIGNALS * N);
    printf("[INIT] Generating input signals (50 × M-sequence)...\n");
    std::vector<int32_t> m_seq = generate_m_sequence(N, 0xABCDEF00);
    for (int i = 0; i < NUM_SIGNALS; i++) {
        std::memcpy(input_signals.data() + i * N, m_seq.data(), N * sizeof(int32_t));
    }
    printf("[OK] Input signals ready\n\n");
    
    double time_upload_input, time_callback_input, time_fft_input;
    profiler.start("Step2_Total");
    
    try {
        fft_handler.step2_input_signals(input_signals.data(), N, NUM_SIGNALS, SCALE_FACTOR, time_upload_input, time_callback_input, time_fft_input);
    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR: Step 2 failed: %s\n", e.what());
        profiler.stop("Step2_Total", Profiler::MILLISECONDS);
        fft_handler.cleanup();
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return;
    }
    
    profiler.stop("Step2_Total", Profiler::MILLISECONDS);
    
    printf("Step 2 Results:\n");
    printf("  Upload:     %.3f ms\n", time_upload_input);
    printf("  FFT (50):   %.3f ms\n", time_fft_input);
    printf("  TOTAL:      %.3f ms\n\n", time_upload_input + time_fft_input);
    
    // ========================================================================
    // 6. STEP 3: КОРРЕЛЯЦИЯ
    // ========================================================================
    
    printf("========== STEP 3 (CORRELATION) ==========\n\n");
    
    double time_multiply, time_ifft, time_download;
    profiler.start("Step3_Total");
    
    try {
        fft_handler.step3_correlation(NUM_SIGNALS, NUM_SHIFTS, N, N_KG, time_multiply, time_ifft, time_download);
    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR: Step 3 failed: %s\n", e.what());
        profiler.stop("Step3_Total", Profiler::MILLISECONDS);
        fft_handler.cleanup();
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return;
    }
    
    profiler.stop("Step3_Total", Profiler::MILLISECONDS);
    
    // ========================================================================
    // 7. ВЫВЕСТИ РЕЗУЛЬТАТЫ
    // ========================================================================
    
    printf("========== STEP 3 RESULTS ==========\n\n");
    
    printf("Timing breakdown:\n");
    printf("  Complex multiply:         %.3f ms\n", time_multiply);
    printf("  Inverse FFT (2000):        %.3f ms\n", time_ifft);
    printf("  Download results:          %.3f ms\n", time_download);
    printf("  ─────────────────────────────────\n");
    printf("  TOTAL:                     %.3f ms\n\n", time_multiply + time_ifft + time_download);
    
    printf("Correlation output:\n");
    printf("  [50][40][5] = %d correlations\n", NUM_SIGNALS * NUM_SHIFTS);
    printf("  Each: peak magnitude + 4 zeros\n");
    printf("  Total output size: %.2f KB\n\n", NUM_SIGNALS * NUM_SHIFTS * N_KG * sizeof(float) / 1024.0f);
    
    printf("========== FULL PIPELINE RESULTS ==========\n\n");
    
    double total_all = (time_upload_ref + time_callback_ref + time_fft_ref) +
                       (time_upload_input + time_callback_input + time_fft_input) +
                       (time_multiply + time_ifft + time_download);
    
    printf("Step 1 (Reference):    %.3f ms\n", time_upload_ref + time_callback_ref + time_fft_ref);
    printf("Step 2 (Input):        %.3f ms\n", time_upload_input + time_callback_input + time_fft_input);
    printf("Step 3 (Correlation):  %.3f ms\n", time_multiply + time_ifft + time_download);
    printf("─────────────────────────────────\n");
    printf("TOTAL PIPELINE:        %.3f ms\n\n", total_all);
    
    printf("Performance:\n");
    printf("  Correlations per ms:   %.2f\n", (NUM_SIGNALS * NUM_SHIFTS) / (time_multiply + time_ifft));
    printf("  FFT size efficiency:   %.2f\n", (N * NUM_SIGNALS * NUM_SHIFTS) / (time_multiply + time_ifft) / 1e6);
    
    printf("\n");
    printf("========== PROFILING STATISTICS ==========\n\n");
    profiler.print_all("FULL PIPELINE PROFILING");
    
    printf("\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("✅ ШАГ 1, 2 & 3 COMPLETE!\n");
    printf("═══════════════════════════════════════════════════════════\n\n");
    
    printf("Summary:\n");
    printf("  ✅ Reference signal processed (40 FFTs)\n");
    printf("  ✅ Input signals processed (50 FFTs)\n");
    printf("  ✅ Correlation computed (2000 correlations)\n");
    printf("  ✅ Results extracted and downloaded\n\n");
    
    // ========================================================================
    // 8. ОЧИСТКА
    // ========================================================================
    
    printf("[GPU] Cleaning up...\n");
    fft_handler.cleanup();
    
    clfftTeardown();
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    printf("[OK] Cleanup complete\n\n");
    
    printf("═══════════════════════════════════════════════════════════\n");
    printf("✨ FFT CORRELATOR PIPELINE COMPLETE! ✨\n");
    printf("═══════════════════════════════════════════════════════════\n\n");
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char* argv[]) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║     FFT CORRELATOR - FULL PIPELINE (STEPS 1, 2 & 3)         ║\n");
    printf("║        Reference → Input → Correlation → Results            ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    printf("Configuration:\n");
    printf("  Signal size (N): 2^15 = %zu\n", N);
    printf("  Num shifts (reference): %d\n", NUM_SHIFTS);
    printf("  Num input signals: %d\n", NUM_SIGNALS);
    printf("  Num output points (n_kg): %d\n", N_KG);
    printf("  Total correlations: %d × %d = %d\n", NUM_SIGNALS, NUM_SHIFTS, NUM_SIGNALS * NUM_SHIFTS);
    printf("  Scale factor: %.2e\n\n", SCALE_FACTOR);
    
    Profiler profiler;
    run_step3(profiler);
    
    return 0;
}
