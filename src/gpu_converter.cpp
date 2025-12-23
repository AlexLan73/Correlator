#include "gpu_converter.hpp"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>

// ============================================================================
// Utility: OpenCL Error Checking
// ============================================================================

const char* get_cl_error_string(cl_int error) {
    switch (error) {
        case CL_SUCCESS:                            return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:                   return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:               return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:             return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:                   return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:                 return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:                   return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:              return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:              return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:                        return "CL_MAP_FAILURE";
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:       return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case CL_COMPILE_PROGRAM_FAILURE:            return "CL_COMPILE_PROGRAM_FAILURE";
        case CL_LINKER_NOT_AVAILABLE:               return "CL_LINKER_NOT_AVAILABLE";
        case CL_LINK_PROGRAM_FAILURE:               return "CL_LINK_PROGRAM_FAILURE";
        case CL_DEVICE_PARTITION_FAILED:            return "CL_DEVICE_PARTITION_FAILED";
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
        case CL_INVALID_VALUE:                      return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:                return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:                   return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:                     return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:                    return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:           return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:              return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:                   return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:                 return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:                 return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:                    return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:                     return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:              return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:                    return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:                return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:          return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:                     return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:                  return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:                  return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:                   return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:                return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:             return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:            return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:             return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:              return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:            return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:                      return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:                  return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT:                  return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE:                return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL:                  return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE:           return "CL_INVALID_GLOBAL_WORK_SIZE";
        case CL_INVALID_PROPERTY:                   return "CL_INVALID_PROPERTY";
        case CL_INVALID_IMAGE_DESCRIPTOR:           return "CL_INVALID_IMAGE_DESCRIPTOR";
        case CL_INVALID_COMPILER_OPTIONS:           return "CL_INVALID_COMPILER_OPTIONS";
        case CL_INVALID_LINKER_OPTIONS:             return "CL_INVALID_LINKER_OPTIONS";
        case CL_INVALID_DEVICE_PARTITION_COUNT:     return "CL_INVALID_DEVICE_PARTITION_COUNT";
        default:                                    return "CL_UNKNOWN_ERROR";
    }
}

#define CHECK_CL_ERROR(err, msg) \
    if (err != CL_SUCCESS) { \
        fprintf(stderr, "ERROR [%s]: %s (code %d)\n", (msg), get_cl_error_string(err), err); \
        return err; \
    }

// ============================================================================
// Initialization
// ============================================================================

cl_int init_gpu_context(
    GPUConverterContext& ctx,
    cl_device_type device_type
) {
    printf("[GPU] Initializing OpenCL context...\n");
    
    cl_int err = CL_SUCCESS;
    cl_platform_id platform_id = nullptr;
    cl_device_id device_id = nullptr;
    
    // Получить платформу
    err = clGetPlatformIDs(1, &platform_id, nullptr);
    CHECK_CL_ERROR(err, "clGetPlatformIDs");
    
    // Получить устройство
    err = clGetDeviceIDs(platform_id, device_type, 1, &device_id, nullptr);
    CHECK_CL_ERROR(err, "clGetDeviceIDs");
    
    ctx.device = device_id;
    
    // Создать контекст
    ctx.context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);
    CHECK_CL_ERROR(err, "clCreateContext");
    
    // Создать очередь (обычная)
    cl_queue_properties queue_props_normal[] = {
        CL_QUEUE_PROPERTIES, 0,
        0
    };
    ctx.queue = clCreateCommandQueueWithProperties(ctx.context, device_id, queue_props_normal, &err);
    CHECK_CL_ERROR(err, "clCreateCommandQueueWithProperties (normal)");
    
    // Создать очередь с профилированием
    cl_queue_properties queue_props_profiling[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
        0
    };
    ctx.profiling_queue = clCreateCommandQueueWithProperties(
        ctx.context, 
        device_id, 
        queue_props_profiling, 
        &err
    );
    CHECK_CL_ERROR(err, "clCreateCommandQueueWithProperties (profiling)");
    
    printf("[OK] GPU context initialized\n");
    return CL_SUCCESS;
}

cl_int load_kernels(
    GPUConverterContext& ctx,
    const std::string& kernel_file
) {
    printf("[GPU] Loading kernels from '%s'...\n", kernel_file.c_str());
    
    cl_int err = CL_SUCCESS;
    
    // Прочитать исходный файл kernel
    std::ifstream file(kernel_file);
    if (!file.is_open()) {
        fprintf(stderr, "ERROR: Cannot open kernel file '%s'\n", kernel_file.c_str());
        return CL_INVALID_VALUE;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string source_code = buffer.str();
    file.close();
    
    printf("[GPU] Kernel source size: %zu bytes\n", source_code.size());
    
    // Создать программу
    const char* source_ptr = source_code.c_str();
    size_t source_len = source_code.length();
    
    cl_program program = clCreateProgramWithSource(
        ctx.context, 
        1, 
        &source_ptr, 
        &source_len, 
        &err
    );
    CHECK_CL_ERROR(err, "clCreateProgramWithSource");
    
    // Скомпилировать программу
    printf("[GPU] Compiling kernels...\n");
    err = clBuildProgram(program, 1, &ctx.device, "", nullptr, nullptr);
    
    if (err != CL_SUCCESS) {
        // Получить лог компиляции
        size_t log_size = 0;
        clGetProgramBuildInfo(program, ctx.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, ctx.device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        
        fprintf(stderr, "COMPILE ERROR:\n%s\n", log.data());
        return err;
    }
    
    printf("[OK] Kernels compiled successfully\n");
    
    // Создать kernel объекты
    printf("[GPU] Creating kernel objects...\n");
    
    ctx.kernel_convert_simple = clCreateKernel(program, "convert_int32_to_float2", &err);
    CHECK_CL_ERROR(err, "clCreateKernel(convert_int32_to_float2)");
    
    ctx.kernel_cyclic_shifts = clCreateKernel(program, "apply_cyclic_shifts", &err);
    CHECK_CL_ERROR(err, "clCreateKernel(apply_cyclic_shifts)");
    
    ctx.kernel_cyclic_shifts_batch = clCreateKernel(program, "apply_cyclic_shifts_batch", &err);
    CHECK_CL_ERROR(err, "clCreateKernel(apply_cyclic_shifts_batch)");
    
    ctx.kernel_fill_test_data = clCreateKernel(program, "fill_test_data", &err);
    CHECK_CL_ERROR(err, "clCreateKernel(fill_test_data)");
    
    // Освободить программу (kernel объекты их будут удерживать)
    clReleaseProgram(program);
    
    printf("[OK] All kernels created successfully\n");
    return CL_SUCCESS;
}

void cleanup_gpu_context(GPUConverterContext& ctx) {
    printf("[GPU] Cleaning up GPU context...\n");
    
    if (ctx.kernel_convert_simple) clReleaseKernel(ctx.kernel_convert_simple);
    if (ctx.kernel_cyclic_shifts) clReleaseKernel(ctx.kernel_cyclic_shifts);
    if (ctx.kernel_cyclic_shifts_batch) clReleaseKernel(ctx.kernel_cyclic_shifts_batch);
    if (ctx.kernel_fill_test_data) clReleaseKernel(ctx.kernel_fill_test_data);
    
    if (ctx.queue) clReleaseCommandQueue(ctx.queue);
    if (ctx.profiling_queue) clReleaseCommandQueue(ctx.profiling_queue);
    if (ctx.context) clReleaseContext(ctx.context);
    
    printf("[OK] GPU context cleaned up\n");
}

// ============================================================================
// GPU Conversion Functions
// ============================================================================

cl_int gpu_convert_simple(
    GPUConverterContext& ctx,
    cl_mem d_input,
    cl_mem d_output,
    size_t num_elements,
    float scale_factor,
    Profiler& profiler,
    const std::string& profile_label,
    cl_event* event
) {
    cl_int err = CL_SUCCESS;
    
    // Установить аргументы kernel
    err = clSetKernelArg(ctx.kernel_convert_simple, 0, sizeof(cl_mem), &d_input);
    CHECK_CL_ERROR(err, "clSetKernelArg(0)");
    
    err = clSetKernelArg(ctx.kernel_convert_simple, 1, sizeof(cl_mem), &d_output);
    CHECK_CL_ERROR(err, "clSetKernelArg(1)");
    
    err = clSetKernelArg(ctx.kernel_convert_simple, 2, sizeof(float), &scale_factor);
    CHECK_CL_ERROR(err, "clSetKernelArg(2)");
    
    unsigned int num_elems_u = static_cast<unsigned int>(num_elements);
    err = clSetKernelArg(ctx.kernel_convert_simple, 3, sizeof(unsigned int), &num_elems_u);
    CHECK_CL_ERROR(err, "clSetKernelArg(3)");
    
    // Определить global work size (один thread на элемент)
    size_t global_work_size = num_elements;
    size_t local_work_size = 256;  // типичный размер для GPU
    
    // Убедиться, что global_work_size кратна local_work_size
    if (global_work_size % local_work_size != 0) {
        global_work_size = ((global_work_size + local_work_size - 1) / local_work_size) * local_work_size;
    }
    
    cl_event kernel_event;
    
    // Запустить kernel на profiling queue
    err = clEnqueueNDRangeKernel(
        ctx.profiling_queue,
        ctx.kernel_convert_simple,
        1,  // 1D kernel
        nullptr,  // global offset
        &global_work_size,
        &local_work_size,
        0, nullptr,  // wait list
        &kernel_event
    );
    CHECK_CL_ERROR(err, "clEnqueueNDRangeKernel(convert_simple)");
    
    // Профилировать
    double elapsed_us = profiler.profile_cl_event(kernel_event, profile_label, Profiler::MICROSECONDS);
    
    if (event) {
        *event = kernel_event;
    } else {
        clReleaseEvent(kernel_event);
    }
    
    return CL_SUCCESS;
}

cl_int gpu_convert_cyclic_shifts(
    GPUConverterContext& ctx,
    cl_mem d_input,
    cl_mem d_output,
    size_t N,
    int num_shifts,
    float scale_factor,
    Profiler& profiler,
    const std::string& profile_label,
    cl_event* event
) {
    cl_int err = CL_SUCCESS;
    
    // Установить аргументы kernel
    err = clSetKernelArg(ctx.kernel_cyclic_shifts, 0, sizeof(cl_mem), &d_input);
    CHECK_CL_ERROR(err, "clSetKernelArg(0)");
    
    err = clSetKernelArg(ctx.kernel_cyclic_shifts, 1, sizeof(cl_mem), &d_output);
    CHECK_CL_ERROR(err, "clSetKernelArg(1)");
    
    err = clSetKernelArg(ctx.kernel_cyclic_shifts, 2, sizeof(float), &scale_factor);
    CHECK_CL_ERROR(err, "clSetKernelArg(2)");
    
    unsigned int N_u = static_cast<unsigned int>(N);
    unsigned int num_shifts_u = static_cast<unsigned int>(num_shifts);
    
    err = clSetKernelArg(ctx.kernel_cyclic_shifts, 3, sizeof(unsigned int), &N_u);
    CHECK_CL_ERROR(err, "clSetKernelArg(3)");
    
    err = clSetKernelArg(ctx.kernel_cyclic_shifts, 4, sizeof(unsigned int), &num_shifts_u);
    CHECK_CL_ERROR(err, "clSetKernelArg(4)");
    
    // Global work size: N * num_shifts (один thread на каждый выходной элемент)
    size_t total_elements = N * num_shifts;
    size_t global_work_size = total_elements;
    size_t local_work_size = 256;
    
    if (global_work_size % local_work_size != 0) {
        global_work_size = ((global_work_size + local_work_size - 1) / local_work_size) * local_work_size;
    }
    
    cl_event kernel_event;
    
    err = clEnqueueNDRangeKernel(
        ctx.profiling_queue,
        ctx.kernel_cyclic_shifts,
        1,
        nullptr,
        &global_work_size,
        &local_work_size,
        0, nullptr,
        &kernel_event
    );
    CHECK_CL_ERROR(err, "clEnqueueNDRangeKernel(cyclic_shifts)");
    
    double elapsed_us = profiler.profile_cl_event(kernel_event, profile_label, Profiler::MICROSECONDS);
    
    if (event) {
        *event = kernel_event;
    } else {
        clReleaseEvent(kernel_event);
    }
    
    return CL_SUCCESS;
}

cl_int gpu_convert_cyclic_shifts_batch(
    GPUConverterContext& ctx,
    cl_mem d_input,
    cl_mem d_output,
    size_t N,
    int shift_start,
    int num_shifts_to_process,
    float scale_factor,
    Profiler& profiler,
    const std::string& profile_label,
    cl_event* event
) {
    cl_int err = CL_SUCCESS;
    
    // Установить аргументы kernel
    err = clSetKernelArg(ctx.kernel_cyclic_shifts_batch, 0, sizeof(cl_mem), &d_input);
    err |= clSetKernelArg(ctx.kernel_cyclic_shifts_batch, 1, sizeof(cl_mem), &d_output);
    err |= clSetKernelArg(ctx.kernel_cyclic_shifts_batch, 2, sizeof(float), &scale_factor);
    
    unsigned int N_u = static_cast<unsigned int>(N);
    unsigned int shift_start_u = static_cast<unsigned int>(shift_start);
    unsigned int num_shifts_u = static_cast<unsigned int>(num_shifts_to_process);
    
    err |= clSetKernelArg(ctx.kernel_cyclic_shifts_batch, 3, sizeof(unsigned int), &N_u);
    err |= clSetKernelArg(ctx.kernel_cyclic_shifts_batch, 4, sizeof(unsigned int), &shift_start_u);
    err |= clSetKernelArg(ctx.kernel_cyclic_shifts_batch, 5, sizeof(unsigned int), &num_shifts_u);
    
    CHECK_CL_ERROR(err, "clSetKernelArg(batch)");
    
    size_t total_elements = N * num_shifts_to_process;
    size_t global_work_size = total_elements;
    size_t local_work_size = 256;
    
    if (global_work_size % local_work_size != 0) {
        global_work_size = ((global_work_size + local_work_size - 1) / local_work_size) * local_work_size;
    }
    
    cl_event kernel_event;
    
    err = clEnqueueNDRangeKernel(
        ctx.profiling_queue,
        ctx.kernel_cyclic_shifts_batch,
        1,
        nullptr,
        &global_work_size,
        &local_work_size,
        0, nullptr,
        &kernel_event
    );
    CHECK_CL_ERROR(err, "clEnqueueNDRangeKernel(cyclic_shifts_batch)");
    
    double elapsed_us = profiler.profile_cl_event(kernel_event, profile_label, Profiler::MICROSECONDS);
    
    if (event) {
        *event = kernel_event;
    } else {
        clReleaseEvent(kernel_event);
    }
    
    return CL_SUCCESS;
}

// ============================================================================
// Utility Functions
// ============================================================================

cl_int gpu_fill_test_data(
    GPUConverterContext& ctx,
    cl_mem d_output,
    size_t num_elements,
    int seed
) {
    cl_int err = CL_SUCCESS;
    
    err = clSetKernelArg(ctx.kernel_fill_test_data, 0, sizeof(cl_mem), &d_output);
    
    unsigned int num_elems_u = static_cast<unsigned int>(num_elements);
    err |= clSetKernelArg(ctx.kernel_fill_test_data, 1, sizeof(unsigned int), &num_elems_u);
    err |= clSetKernelArg(ctx.kernel_fill_test_data, 2, sizeof(int), &seed);
    
    CHECK_CL_ERROR(err, "clSetKernelArg(fill_test_data)");
    
    size_t global_work_size = num_elements;
    size_t local_work_size = 256;
    
    if (global_work_size % local_work_size != 0) {
        global_work_size = ((global_work_size + local_work_size - 1) / local_work_size) * local_work_size;
    }
    
    err = clEnqueueNDRangeKernel(
        ctx.queue,
        ctx.kernel_fill_test_data,
        1,
        nullptr,
        &global_work_size,
        &local_work_size,
        0, nullptr,
        nullptr
    );
    CHECK_CL_ERROR(err, "clEnqueueNDRangeKernel(fill_test_data)");
    
    clFinish(ctx.queue);
    return CL_SUCCESS;
}

cl_int gpu_copy_to_device(
    GPUConverterContext& ctx,
    const void* host_data,
    cl_mem device_buffer,
    size_t size,
    cl_event* event
) {
    cl_int err = clEnqueueWriteBuffer(
        ctx.queue,
        device_buffer,
        CL_TRUE,  // blocking
        0,
        size,
        host_data,
        0, nullptr,
        event
    );
    CHECK_CL_ERROR(err, "clEnqueueWriteBuffer");
    return CL_SUCCESS;
}

cl_int gpu_copy_from_device(
    GPUConverterContext& ctx,
    cl_mem device_buffer,
    void* host_data,
    size_t size,
    cl_event* event
) {
    cl_int err = clEnqueueReadBuffer(
        ctx.queue,
        device_buffer,
        CL_TRUE,  // blocking
        0,
        size,
        host_data,
        0, nullptr,
        event
    );
    CHECK_CL_ERROR(err, "clEnqueueReadBuffer");
    return CL_SUCCESS;
}

void print_gpu_info(const GPUConverterContext& ctx) {
    char device_name[1024];
    char device_vendor[1024];
    cl_uint compute_units = 0;
    size_t max_work_group_size = 0;
    
    clGetDeviceInfo(ctx.device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    clGetDeviceInfo(ctx.device, CL_DEVICE_VENDOR, sizeof(device_vendor), device_vendor, nullptr);
    clGetDeviceInfo(ctx.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, nullptr);
    clGetDeviceInfo(ctx.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, nullptr);
    
    printf("\n========== GPU INFO ==========\n");
    printf("Device Name:          %s\n", device_name);
    printf("Device Vendor:        %s\n", device_vendor);
    printf("Compute Units:        %u\n", compute_units);
    printf("Max Work Group Size:  %zu\n", max_work_group_size);
    printf("==============================\n\n");
}

size_t get_max_work_group_size(
    GPUConverterContext& ctx,
    cl_kernel kernel
) {
    size_t max_work_group_size = 0;
    clGetKernelWorkGroupInfo(
        kernel,
        ctx.device,
        CL_KERNEL_WORK_GROUP_SIZE,
        sizeof(max_work_group_size),
        &max_work_group_size,
        nullptr
    );
    return max_work_group_size;
}

void benchmark_gpu_conversion(
    GPUConverterContext& ctx,
    Profiler& profiler,
    int num_runs
) {
    printf("\n========== GPU CONVERSION BENCHMARK ==========\n");
    printf("Running %d iterations...\n\n", num_runs);
    
    // Параметры
    const size_t N = (1 << 15);
    const float scale_factor = 1.0f / 32768.0f;
    
    // Выделить GPU память
    cl_int err;
    cl_mem d_input = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY, N * sizeof(int), nullptr, &err);
    cl_mem d_output = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY, N * sizeof(cl_float2), nullptr, &err);
    
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: Cannot allocate GPU memory\n");
        return;
    }
    
    // Заполнить тестовыми данными
    gpu_fill_test_data(ctx, d_input, N, 42);
    
    printf("Testing simple conversion (N=%zu):\n", N);
    for (int i = 0; i < num_runs; i++) {
        gpu_convert_simple(ctx, d_input, d_output, N, scale_factor, profiler, "GPU_convert_simple");
    }
    
    profiler.print("GPU_convert_simple");
    printf("\n");
    
    // Очистить
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
}
