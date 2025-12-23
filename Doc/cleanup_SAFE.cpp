// ============================================================================
// ИСПРАВЛЕННЫЙ cleanup() - ЗАЩИТА ОТ ДВОЙНОГО ВЫЗОВА
// ============================================================================

void FFTHandler::cleanup() {
    // ✅ ЗАЩИТА: Если уже вычищено - не трогаем!
    if (ctx_.is_cleaned_up) {
        printf("[FFT] Already cleaned up, skipping...\n");
        return;
    }
    
    // ✅ ЗАЩИТА: Если вообще не инициализировано - не трогаем!
    if (!ctx_.initialized) {
        printf("[FFT] Not initialized, skipping cleanup\n");
        return;
    }
    
    printf("[FFT] Cleaning up GPU resources...\n");
    
    // ========================================================================
    // 1. DESTROY FFT PLANS FIRST (ВАЖНО!)
    // ========================================================================
    
    printf("  1. Destroying FFT plans...\n");
    
    if (ctx_.reference_fft_plan) {
        clfftStatus status = clfftDestroyPlan(&ctx_.reference_fft_plan);
        if (status == CLFFT_SUCCESS) {
            printf("     ✓ Reference FFT plan destroyed\n");
        } else {
            printf("     ✗ Failed to destroy reference FFT plan (code: %d)\n", status);
        }
        ctx_.reference_fft_plan = nullptr;
    }
    
    if (ctx_.input_fft_plan) {
        clfftStatus status = clfftDestroyPlan(&ctx_.input_fft_plan);
        if (status == CLFFT_SUCCESS) {
            printf("     ✓ Input FFT plan destroyed\n");
        } else {
            printf("     ✗ Failed to destroy input FFT plan (code: %d)\n", status);
        }
        ctx_.input_fft_plan = nullptr;
    }
    
    if (ctx_.correlation_ifft_plan) {
        clfftStatus status = clfftDestroyPlan(&ctx_.correlation_ifft_plan);
        if (status == CLFFT_SUCCESS) {
            printf("     ✓ Correlation IFFT plan destroyed\n");
        } else {
            printf("     ✗ Failed to destroy correlation IFFT plan (code: %d)\n", status);
        }
        ctx_.correlation_ifft_plan = nullptr;
    }
    
    // ========================================================================
    // 2. RELEASE GPU MEMORY BUFFERS (После разрушения планов!)
    // ========================================================================
    
    printf("  2. Releasing GPU memory buffers...\n");
    
    if (ctx_.reference_data) {
        cl_int status = clReleaseMemObject(ctx_.reference_data);
        if (status == CL_SUCCESS) {
            printf("     ✓ Reference data buffer released\n");
        } else {
            printf("     ✗ Failed to release reference data (code: %d)\n", status);
        }
        ctx_.reference_data = nullptr;
    }
    
    if (ctx_.reference_fft) {
        cl_int status = clReleaseMemObject(ctx_.reference_fft);
        if (status == CL_SUCCESS) {
            printf("     ✓ Reference FFT buffer released\n");
        } else {
            printf("     ✗ Failed to release reference FFT (code: %d)\n", status);
        }
        ctx_.reference_fft = nullptr;
    }
    
    if (ctx_.input_data) {
        cl_int status = clReleaseMemObject(ctx_.input_data);
        if (status == CL_SUCCESS) {
            printf("     ✓ Input data buffer released\n");
        } else {
            printf("     ✗ Failed to release input data (code: %d)\n", status);
        }
        ctx_.input_data = nullptr;
    }
    
    if (ctx_.input_fft) {
        cl_int status = clReleaseMemObject(ctx_.input_fft);
        if (status == CL_SUCCESS) {
            printf("     ✓ Input FFT buffer released\n");
        } else {
            printf("     ✗ Failed to release input FFT (code: %d)\n", status);
        }
        ctx_.input_fft = nullptr;
    }
    
    if (ctx_.correlation_fft) {
        cl_int status = clReleaseMemObject(ctx_.correlation_fft);
        if (status == CL_SUCCESS) {
            printf("     ✓ Correlation FFT buffer released\n");
        } else {
            printf("     ✗ Failed to release correlation FFT (code: %d)\n", status);
        }
        ctx_.correlation_fft = nullptr;
    }
    
    if (ctx_.correlation_ifft) {
        cl_int status = clReleaseMemObject(ctx_.correlation_ifft);
        if (status == CL_SUCCESS) {
            printf("     ✓ Correlation IFFT buffer released\n");
        } else {
            printf("     ✗ Failed to release correlation IFFT (code: %d)\n", status);
        }
        ctx_.correlation_ifft = nullptr;
    }
    
    if (ctx_.pre_callback_userdata) {
        cl_int status = clReleaseMemObject(ctx_.pre_callback_userdata);
        if (status == CL_SUCCESS) {
            printf("     ✓ Pre-callback userdata buffer released\n");
        } else {
            printf("     ✗ Failed to release pre-callback userdata (code: %d)\n", status);
        }
        ctx_.pre_callback_userdata = nullptr;
    }
    
    if (ctx_.post_callback_userdata) {
        cl_int status = clReleaseMemObject(ctx_.post_callback_userdata);
        if (status == CL_SUCCESS) {
            printf("     ✓ Post-callback userdata buffer released\n");
        } else {
            printf("     ✗ Failed to release post-callback userdata (code: %d)\n", status);
        }
        ctx_.post_callback_userdata = nullptr;
    }
    
    // ========================================================================
    // 3. MARK AS CLEANED UP (ВАЖНО!)
    // ========================================================================
    
    ctx_.initialized = false;
    ctx_.is_cleaned_up = true;  // ← СТАВИМ ФЛАГ!
    
    printf("[OK] GPU cleanup complete!\n\n");
}
