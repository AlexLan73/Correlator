#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Макросы для красивого вывода
#define GREEN "\x1b[32m"
#define RED "\x1b[31m"
#define YELLOW "\x1b[33m"
#define CYAN "\x1b[36m"
#define RESET "\x1b[0m"

void printDeviceInfo(cl_device_id device) {
    char device_name[256];
    char device_vendor[256];
    cl_device_type device_type;
    cl_uint compute_units;
    cl_ulong global_mem;
    cl_uint max_work_group_size;
    
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(device_vendor), device_vendor, NULL);
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
    
    printf("\n" CYAN "═══════════════════════════════════════════════════════" RESET "\n");
    printf(CYAN "📊 DEVICE INFO" RESET "\n");
    printf(CYAN "═══════════════════════════════════════════════════════" RESET "\n");
    printf("Name:              %s\n", device_name);
    printf("Vendor:            %s\n", device_vendor);
    printf("Type:              %s\n", device_type == CL_DEVICE_TYPE_GPU ? "GPU" : 
                                       device_type == CL_DEVICE_TYPE_CPU ? "CPU" : "OTHER");
    printf("Compute Units:     %u\n", compute_units);
    printf("Global Memory:     %llu MB\n", global_mem / (1024 * 1024));
    printf("Max Work Group:    %u\n", max_work_group_size);
}

// ✅ ПРОВЕРКА OpenCL VERSION
void checkOpenCLVersion(cl_device_id device) {
    char version[256];
    clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(version), version, NULL);
    
    printf("\n" CYAN "🔍 OpenCL VERSION" RESET "\n");
    printf(CYAN "─────────────────────────────────────────────────────" RESET "\n");
    printf("Device Version: %s\n", version);
    
    // Парсим версию (формат: "OpenCL X.Y ...")
    int major = 0, minor = 0;
    sscanf(version, "OpenCL %d.%d", &major, &minor);
    
    printf("Parsed Version:  %d.%d\n", major, minor);
    
    if (major >= 3) {
        printf(GREEN "✅ OpenCL 3.0+ ПОДДЕРЖИВАЕТСЯ!" RESET "\n");
    } else if (major == 2) {
        printf(YELLOW "⚠️  OpenCL 2.0 (SVM поддерживается, Command Buffers - НЕТ)" RESET "\n");
    } else {
        printf(RED "❌ OpenCL 1.x (старая версия)" RESET "\n");
    }
}

// ✅ ПРОВЕРКА РАСШИРЕНИЙ
void checkExtensions(cl_device_id device) {
    char extensions[4096];
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(extensions), extensions, NULL);
    
    printf("\n" CYAN "📦 EXTENSIONS" RESET "\n");
    printf(CYAN "─────────────────────────────────────────────────────" RESET "\n");
    
    // Проверяем ключевые расширения
    const char* required_extensions[] = {
        "cl_khr_command_buffer",           // ← Command Buffers!
        "cl_khr_svm",                      // ← Shared Virtual Memory
        "cl_ext_device_fission",
        "cl_nv_device_attribute_query",    // NVIDIA specific
        "cl_amd_device_attribute_query"    // AMD specific
    };
    
    int num_extensions = sizeof(required_extensions) / sizeof(required_extensions[0]);
    
    for (int i = 0; i < num_extensions; i++) {
        if (strstr(extensions, required_extensions[i])) {
            printf(GREEN "✅ %s" RESET "\n", required_extensions[i]);
        } else {
            printf(RED "❌ %s" RESET "\n", required_extensions[i]);
        }
    }
}

// ✅ ПРОВЕРКА SVM ПОДДЕРЖКИ
void checkSVMSupport(cl_device_id device) {
    printf("\n" CYAN "💾 SHARED VIRTUAL MEMORY (SVM) SUPPORT" RESET "\n");
    printf(CYAN "─────────────────────────────────────────────────────" RESET "\n");
    
    cl_device_svm_capabilities svm_caps;
    clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES, sizeof(svm_caps), &svm_caps, NULL);
    
    if (svm_caps == 0) {
        printf(RED "❌ No SVM support" RESET "\n");
        return;
    }
    
    if (svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) {
        printf(YELLOW "⚠️  CL_DEVICE_SVM_COARSE_GRAIN_BUFFER (limited)" RESET "\n");
    }
    
    if (svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) {
        printf(GREEN "✅ CL_DEVICE_SVM_FINE_GRAIN_BUFFER (good!)" RESET "\n");
    }
    
    if (svm_caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) {
        printf(GREEN "✅ CL_DEVICE_SVM_FINE_GRAIN_SYSTEM (excellent!)" RESET "\n");
    }
    
    if (svm_caps & CL_DEVICE_SVM_ATOMICS) {
        printf(GREEN "✅ CL_DEVICE_SVM_ATOMICS" RESET "\n");
    }
}

// ✅ ПРОВЕРКА COMMAND BUFFERS ПОДДЕРЖКИ
void checkCommandBuffersSupport(cl_device_id device) {
    printf("\n" CYAN "🎯 COMMAND BUFFERS (OpenCL 3.0) SUPPORT" RESET "\n");
    printf(CYAN "─────────────────────────────────────────────────────" RESET "\n");
    
    char extensions[4096];
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(extensions), extensions, NULL);
    
    if (strstr(extensions, "cl_khr_command_buffer")) {
        printf(GREEN "✅ cl_khr_command_buffer EXTENSION FOUND!" RESET "\n");
        printf(GREEN "✅ COMMAND BUFFERS ARE SUPPORTED!" RESET "\n");
        
        // Дополнительная информация
        cl_bool command_buffer_capable;
        clGetDeviceInfo(device, CL_DEVICE_COMMAND_BUFFER_CAPABLE_KHR, 
                       sizeof(command_buffer_capable), &command_buffer_capable, NULL);
        
        if (command_buffer_capable) {
            printf(GREEN "✅ Device is command buffer capable!" RESET "\n");
        }
    } else {
        printf(RED "❌ cl_khr_command_buffer NOT FOUND" RESET "\n");
        printf(RED "❌ COMMAND BUFFERS ARE NOT SUPPORTED" RESET "\n");
        printf(YELLOW "\nℹ️  Try alternative: Asynchronous queues without clWaitForEvents()" RESET "\n");
    }
}

// ✅ ПРОВЕРКА UNIFIED MEMORY
void checkUnifiedMemory(cl_device_id device) {
    printf("\n" CYAN "🔗 UNIFIED MEMORY SUPPORT" RESET "\n");
    printf(CYAN "─────────────────────────────────────────────────────" RESET "\n");
    
    cl_bool unified_memory;
    cl_int err = clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY, 
                                 sizeof(unified_memory), &unified_memory, NULL);
    
    if (err == CL_SUCCESS) {
        if (unified_memory) {
            printf(GREEN "✅ Host Unified Memory is SUPPORTED" RESET "\n");
        } else {
            printf(YELLOW "⚠️  Host Unified Memory is NOT supported" RESET "\n");
        }
    }
}

// ✅ РЕКОМЕНДАЦИИ
void printRecommendations(cl_device_id device) {
    char version[256];
    char extensions[4096];
    
    clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(version), version, NULL);
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(extensions), extensions, NULL);
    
    int major = 0, minor = 0;
    sscanf(version, "OpenCL %d.%d", &major, &minor);
    
    printf("\n" CYAN "🎯 RECOMMENDATIONS FOR YOUR ALGORITHM" RESET "\n");
    printf(CYAN "═══════════════════════════════════════════════════════" RESET "\n");
    
    if (major >= 3 && strstr(extensions, "cl_khr_command_buffer")) {
        printf(GREEN "✅ OPTION 1 (BEST): Use Command Buffers!" RESET "\n");
        printf("   - Build 2 command buffers: STEP1, STEP2+3\n");
        printf("   - Expected speedup: 10-25%%\n");
        printf("   - Code example in next section\n\n");
    }
    
    if (major >= 2) {
        printf(YELLOW "✅ OPTION 2: Use Asynchronous Queues + SVM" RESET "\n");
        printf("   - Skip clWaitForEvents() between steps\n");
        printf("   - Use SVM for fine-grain memory management\n");
        printf("   - Expected speedup: 5-15%%\n\n");
    }
    
    printf(YELLOW "✅ OPTION 3 (FALLBACK): Asynchronous Enqueue" RESET "\n");
    printf("   - Use clEnqueueWriteBuffer() without waiting\n");
    printf("   - Call clFinish() only at the end\n");
    printf("   - Expected speedup: 2-5%%\n");
}

// ГЛАВНАЯ ФУНКЦИЯ
int main() {
    printf("\n");
    printf(GREEN "╔════════════════════════════════════════════════════════╗" RESET "\n");
    printf(GREEN "║  OpenCL DEVICE CAPABILITY CHECKER FOR CORRELATOR       ║" RESET "\n");
    printf(GREEN "║  Проверка поддержки Command Buffers и SVM              ║" RESET "\n");
    printf(GREEN "╚════════════════════════════════════════════════════════╝" RESET "\n");
    
    cl_int err;
    
    // ===== ПОЛУЧИТЬ ПЛАТФОРМЫ =====
    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, NULL, &num_platforms);
    
    if (num_platforms == 0) {
        printf(RED "❌ No OpenCL platforms found!" RESET "\n");
        return 1;
    }
    
    cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    clGetPlatformIDs(num_platforms, platforms, NULL);
    
    printf("\nFound %u platform(s)\n\n", num_platforms);
    
    // ===== ДЛЯ КАЖДОЙ ПЛАТФОРМЫ =====
    for (cl_uint p = 0; p < num_platforms; p++) {
        char platform_name[256];
        char platform_vendor[256];
        
        clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
        clGetPlatformInfo(platforms[p], CL_PLATFORM_VENDOR, sizeof(platform_vendor), platform_vendor, NULL);
        
        printf(GREEN "\n╔════════════════════════════════════════════════════════╗" RESET "\n");
        printf(GREEN "║ PLATFORM %u: %s (%s)" RESET "\n", p, platform_name, platform_vendor);
        printf(GREEN "╚════════════════════════════════════════════════════════╝" RESET "\n");
        
        // Получить устройства
        cl_uint num_devices = 0;
        clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        
        if (num_devices == 0) {
            printf(YELLOW "No devices found on this platform\n" RESET);
            continue;
        }
        
        cl_device_id* devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
        clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
        
        // ===== ДЛЯ КАЖДОГО УСТРОЙСТВА =====
        for (cl_uint d = 0; d < num_devices; d++) {
            printf("\n" GREEN "────────────────────────────────────────────────────────" RESET "\n");
            printf(GREEN "DEVICE %u" RESET "\n", d);
            printf(GREEN "────────────────────────────────────────────────────────" RESET "\n");
            
            // Основная информация
            printDeviceInfo(devices[d]);
            
            // Версия OpenCL
            checkOpenCLVersion(devices[d]);
            
            // Расширения
            checkExtensions(devices[d]);
            
            // SVM поддержка
            checkSVMSupport(devices[d]);
            
            // Command Buffers поддержка
            checkCommandBuffersSupport(devices[d]);
            
            // Unified Memory
            checkUnifiedMemory(devices[d]);
            
            // Рекомендации
            printRecommendations(devices[d]);
        }
        
        free(devices);
    }
    
    free(platforms);
    
    // ===== ИТОГОВЫЙ ВЫВОД =====
    printf("\n" GREEN "╔════════════════════════════════════════════════════════╗" RESET "\n");
    printf(GREEN "║  SUMMARY AND NEXT STEPS                                ║" RESET "\n");
    printf(GREEN "╚════════════════════════════════════════════════════════╝" RESET "\n");
    
    printf("\n" CYAN "📝 NEXT STEPS:" RESET "\n");
    printf("1. Check which devices support Command Buffers (cl_khr_command_buffer)\n");
    printf("2. If supported: Implement 2-buffer approach (STEP1, STEP2+3)\n");
    printf("3. If not supported: Use asynchronous enqueue without clWaitForEvents()\n");
    printf("4. Expected speedup with Command Buffers: 10-25%%\n");
    
    printf("\n" CYAN "💡 FOR YOUR RTX 2080 Ti / RTX 3060:" RESET "\n");
    printf("- RTX 2080 Ti likely supports OpenCL 1.2 (may not have CB)\n");
    printf("- RTX 3060 supports OpenCL 1.2 (may not have CB)\n");
    printf("- Fallback: Use async approach for 2-5%% speedup\n");
    
    printf("\n" GREEN "✅ Test completed!" RESET "\n\n");
    
    return 0;
}

