# GDB configuration for Correlator project

# Enable pretty printing
set print pretty on
set print array on
set print array-indexes on

# Set disassembly flavor to Intel
set disassembly-flavor intel

# Handle SIGPIPE
handle SIGPIPE nostop noprint pass

# Set breakpoints for common crash points
# break main
# break FFTHandler::initialize
# break step2_input_signals

# Commands to run after startup
define debug_correlator
    set args
    break main
    run
end

# Show backtrace on crash
define bt_full
    bt full
end

# Print OpenCL error codes
define cl_error
    printf "CL_SUCCESS = 0\n"
    printf "CL_DEVICE_NOT_FOUND = -1\n"
    printf "CL_DEVICE_NOT_AVAILABLE = -2\n"
    printf "CL_COMPILER_NOT_AVAILABLE = -3\n"
    printf "CL_MEM_OBJECT_ALLOCATION_FAILURE = -4\n"
    printf "CL_OUT_OF_RESOURCES = -5\n"
    printf "CL_OUT_OF_HOST_MEMORY = -6\n"
    printf "CL_PROFILING_INFO_NOT_AVAILABLE = -7\n"
    printf "CL_MEM_COPY_OVERLAP = -8\n"
    printf "CL_IMAGE_FORMAT_MISMATCH = -9\n"
    printf "CL_IMAGE_FORMAT_NOT_SUPPORTED = -10\n"
    printf "CL_BUILD_PROGRAM_FAILURE = -11\n"
    printf "CL_MAP_FAILURE = -12\n"
    printf "CL_INVALID_VALUE = -30\n"
    printf "CL_INVALID_DEVICE_TYPE = -31\n"
    printf "CL_INVALID_PLATFORM = -32\n"
    printf "CL_INVALID_DEVICE = -33\n"
    printf "CL_INVALID_CONTEXT = -34\n"
    printf "CL_INVALID_QUEUE_PROPERTIES = -35\n"
    printf "CL_INVALID_COMMAND_QUEUE = -36\n"
    printf "CL_INVALID_HOST_PTR = -37\n"
    printf "CL_INVALID_MEM_OBJECT = -38\n"
    printf "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR = -39\n"
    printf "CL_INVALID_IMAGE_SIZE = -40\n"
    printf "CL_INVALID_SAMPLER = -41\n"
    printf "CL_INVALID_BINARY = -42\n"
    printf "CL_INVALID_BUILD_OPTIONS = -43\n"
    printf "CL_INVALID_PROGRAM = -44\n"
    printf "CL_INVALID_PROGRAM_EXECUTABLE = -45\n"
    printf "CL_INVALID_KERNEL_NAME = -46\n"
    printf "CL_INVALID_KERNEL_DEFINITION = -47\n"
    printf "CL_INVALID_KERNEL = -48\n"
    printf "CL_INVALID_ARG_INDEX = -49\n"
    printf "CL_INVALID_ARG_VALUE = -50\n"
    printf "CL_INVALID_ARG_SIZE = -51\n"
    printf "CL_INVALID_KERNEL_ARGS = -52\n"
    printf "CL_INVALID_WORK_DIMENSION = -53\n"
    printf "CL_INVALID_WORK_GROUP_SIZE = -54\n"
    printf "CL_INVALID_WORK_ITEM_SIZE = -55\n"
    printf "CL_INVALID_GLOBAL_OFFSET = -56\n"
    printf "CL_INVALID_EVENT_WAIT_LIST = -57\n"
    printf "CL_INVALID_EVENT = -58\n"
    printf "CL_INVALID_OPERATION = -59\n"
    printf "CL_INVALID_GL_OBJECT = -60\n"
    printf "CL_INVALID_BUFFER_SIZE = -61\n"
    printf "CL_INVALID_MIP_LEVEL = -62\n"
end

