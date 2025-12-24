#ifndef DEBUG_LOG_HPP
#define DEBUG_LOG_HPP

#include <cstdio>
#include <iostream>
#include <sstream>

// ============================================================================
// Debug Output Control
// ============================================================================

// Определяем режим сборки
#ifndef NDEBUG
    // Debug режим - включен вывод
    #define DEBUG_OUTPUT 1
    #define RELEASE_BUILD 0
#else
    // Release режим - отключен вывод
    #define DEBUG_OUTPUT 0
    #define RELEASE_BUILD 1
#endif

// Альтернативная проверка для Visual Studio
#ifdef _DEBUG
    #undef DEBUG_OUTPUT
    #define DEBUG_OUTPUT 1
    #undef RELEASE_BUILD
    #define RELEASE_BUILD 0
#endif

// ============================================================================
// Макросы для условного вывода
// ============================================================================

/**
 * DEBUG_LOG - для отладочных сообщений (только в Debug режиме)
 * Использование: DEBUG_LOG("Step 1: Processing...\n");
 */
#if DEBUG_OUTPUT
    #define DEBUG_LOG(...) do { \
        std::printf(__VA_ARGS__); \
    } while(0)
#else
    #define DEBUG_LOG(...) ((void)0)
#endif

/**
 * VERBOSE_LOG - для детальных отладочных сообщений
 * Можно отключить даже в Debug через VERBOSE_DEBUG
 */
#ifndef VERBOSE_DEBUG
    #define VERBOSE_DEBUG DEBUG_OUTPUT
#endif

#if VERBOSE_DEBUG
    #define VERBOSE_LOG(...) do { \
        std::printf(__VA_ARGS__); \
    } while(0)
#else
    #define VERBOSE_LOG(...) ((void)0)
#endif

/**
 * INFO_LOG - для информационных сообщений
 * В Release режиме отключен (можно включить через RELEASE_VERBOSE=1)
 */
#ifndef RELEASE_VERBOSE
    #define RELEASE_VERBOSE 0
#endif

#if DEBUG_OUTPUT || RELEASE_VERBOSE
    #define INFO_LOG(...) do { \
        std::printf(__VA_ARGS__); \
    } while(0)
#else
    #define INFO_LOG(...) ((void)0)
#endif

/**
 * ERROR_LOG - для сообщений об ошибках (всегда включен)
 * Используется для критических ошибок
 */
#define ERROR_LOG(...) do { \
    std::fprintf(stderr, __VA_ARGS__); \
} while(0)

/**
 * WARNING_LOG - для предупреждений (всегда включен)
 */
#define WARNING_LOG(...) do { \
    std::fprintf(stderr, "WARNING: "); \
    std::fprintf(stderr, __VA_ARGS__); \
} while(0)

/**
 * COUT_LOG - для вывода через std::cout (условный, как INFO_LOG)
 * Используется для пользовательского интерфейса
 */
#if DEBUG_OUTPUT || RELEASE_VERBOSE
    #define COUT_LOG(msg) do { \
        std::cout << msg; \
    } while(0)
#else
    #define COUT_LOG(msg) ((void)0)
#endif

// ============================================================================
// Проверка режима сборки (для отладки)
// ============================================================================

inline void print_build_info() {
    #if DEBUG_OUTPUT
        INFO_LOG("[BUILD] Debug mode: DEBUG_OUTPUT=1\n");
    #else
        INFO_LOG("[BUILD] Release mode: DEBUG_OUTPUT=0\n");
    #endif
}

#endif // DEBUG_LOG_HPP


