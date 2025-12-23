-- ============================================================================
-- ЛОКАЛЬНАЯ ПАМЯТЬ: SQLite схема для проекта AmdOpenCLTest01
-- ============================================================================
-- Описание: База данных для хранения специфичных для проекта решений
-- Цель: Сохранение архитектурных решений и результатов бенчмарков
-- ============================================================================

-- Таблица архитектурных решений проекта
CREATE TABLE IF NOT EXISTS project_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context TEXT NOT NULL,            -- 'nvidia_backend', 'opencl_integration', 'vulkan_setup'
    decision TEXT NOT NULL,           -- Принятое решение
    reasoning TEXT NOT NULL,          -- Обоснование решения
    alternatives_considered TEXT,     -- JSON массив рассмотренных альтернатив
    results TEXT,                     -- JSON с результатами
    impact TEXT,                      -- Влияние на проект
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Таблица результатов бенчмарков
CREATE TABLE IF NOT EXISTS benchmark_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_name TEXT NOT NULL,          -- 'fft16_cuda_custom', 'fft256_opencl_library'
    fft_size INTEGER NOT NULL,        -- Размер FFT
    backend TEXT NOT NULL,            -- 'cuda', 'opencl', 'vulkan', 'cpu'
    kernel_type TEXT NOT NULL,        -- 'custom', 'library', 'fused'
    performance_tflops REAL,          -- Производительность в TFLOP/s
    memory_bandwidth_gbps REAL,       -- Пропускная способность памяти
    upload_time_ms REAL,              -- Время загрузки данных
    compute_time_ms REAL,             -- Время вычислений
    download_time_ms REAL,            -- Время выгрузки данных
    total_time_ms REAL,               -- Общее время
    gpu_utilization REAL,             -- Использование GPU (%)
    memory_usage_mb INTEGER,          -- Использование памяти
    test_parameters TEXT,             -- JSON с параметрами теста
    test_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT                        -- Дополнительные заметки
);

-- Таблица проблем и их решений
CREATE TABLE IF NOT EXISTS problem_solutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    problem_type TEXT NOT NULL,       -- 'compilation', 'runtime', 'performance', 'integration'
    problem_description TEXT NOT NULL, -- Описание проблемы
    error_message TEXT,               -- Текст ошибки
    solution TEXT NOT NULL,           -- Решение
    solution_steps TEXT,              -- JSON массив шагов решения
    time_to_solve_minutes INTEGER,    -- Время на решение (минуты)
    difficulty_level INTEGER,         -- Уровень сложности (1-5)
    related_files TEXT,               -- JSON массив связанных файлов
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Таблица эволюции проекта
CREATE TABLE IF NOT EXISTS project_evolution (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    milestone TEXT NOT NULL,          -- 'initial_setup', 'cuda_integration', 'opencl_working'
    description TEXT NOT NULL,        -- Описание этапа
    achievements TEXT,                -- JSON массив достижений
    challenges TEXT,                  -- JSON массив вызовов
    lessons_learned TEXT,             -- Уроки
    next_steps TEXT,                  -- Следующие шаги
    milestone_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Таблица конфигураций сборки
CREATE TABLE IF NOT EXISTS build_configurations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_name TEXT NOT NULL,        -- 'cpu_only', 'cuda_enabled', 'full_gpu'
    cmake_flags TEXT,                 -- JSON с флагами CMake
    enabled_backends TEXT,            -- JSON массив включенных бэкендов
    compilation_success BOOLEAN,      -- Успешность компиляции
    build_time_seconds INTEGER,       -- Время сборки
    binary_size_mb REAL,              -- Размер бинарника
    dependencies TEXT,                -- JSON с зависимостями
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Таблица тестов и их результатов
CREATE TABLE IF NOT EXISTS test_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_name TEXT NOT NULL,          -- 'fft16_correctness', 'hemming_window', 'sliding_fft'
    test_type TEXT NOT NULL,          -- 'correctness', 'performance', 'integration'
    status TEXT NOT NULL,             -- 'PASS', 'FAIL', 'SKIP'
    execution_time_ms REAL,           -- Время выполнения
    error_message TEXT,               -- Сообщение об ошибке
    test_output TEXT,                 -- Вывод теста
    test_parameters TEXT,             -- JSON с параметрами
    test_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Индексы для быстрого поиска
CREATE INDEX IF NOT EXISTS idx_project_decisions_context ON project_decisions(context);
CREATE INDEX IF NOT EXISTS idx_benchmark_fft_backend ON benchmark_results(fft_size, backend);
CREATE INDEX IF NOT EXISTS idx_problem_type ON problem_solutions(problem_type);
CREATE INDEX IF NOT EXISTS idx_test_name_status ON test_results(test_name, status);

-- Триггеры для обновления updated_at
CREATE TRIGGER IF NOT EXISTS update_project_decisions_timestamp 
    AFTER UPDATE ON project_decisions
    BEGIN
        UPDATE project_decisions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;

