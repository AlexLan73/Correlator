# 🔑 Key Findings - MemoryBank

## 🎯 Основные директории и компоненты

## 🚀 FFT Optimization (2024)
- **Среднее ускорение**: 91.87x vs cuFFT
- **FFT16**: 318x speedup
- **FFT32**: 220x speedup
- **Пиковая производительность**: 2.23 TFLOP/s при 262k-524k окон

## 🎯 GPU Occupancy Analysis
- **Оптимальный batch size**: 8k-64k окон
- **Пик производительности**: 262k-524k окон
- **Критический фактор**: Memory bandwidth vs batch size

## 🏗️ Correlation Architecture
- **Оптимизированный режим**: BATCHED MODE
- **Производительность**: 38.5x vs стандартный режим
- **Kernel launches**: 2 потока по 40
- **Memory efficiency**: Предварительное выделение для FFT

## 🎯 Sliding FFT16 Analysis (2025-10-14)
- **Оптимальный режим**: Все 4 этапа gather → FFT16 → fftshift → store в одном kernel
- **При N < 64 окон**: Предварительное выделение, FFT16, fftshift, store
- **Формула**: s=2*w; x[t]=(s+t<S)?X[1][s+t]:0; FFT16(x)=y; yshift[k]=y[(k+8) mod 16]; B[L][w][k]=yshift[k]
- **При N≥64 окон**: cuFFT и fftshift в отдельных kernel

## 🎯 Current FFT Architecture (2025-10-19)

### 🚀 Supported Platforms
- **Платформы**: NVIDIA, OpenCL, Vulkan, ROCm
- **ROCm поддержка**: Создан для AMD GPU с rocFFT, rocBLAS
- **Производительность**: Сравнение с NVIDIA
- **JSON отчеты**: Автоматизированные результаты тестов

### 🎯 Цели проекта
- **FFT256 > 1.0 TFLOP/s** на всех платформах
- **Нативные реализации** сопоставимы с библиотечными
- **Параллельные вычисления**: 2×FFT512 + 1×FFT1024 в блоке 1024
- **Sliding FFT**: С окнами Хемминга, размеры FFT16 до FFT32768

### 🎯 GitHub MCP Integration

#### 🚀 Настройка для GitHub MCP
- **Аутентификация**: Использовать Personal Access Token (PAT) для доступа к private repo, workflow, write:packages, delete:packages, admin:org, gist, notifications, user, delete_repo
- **Основные функции**: mcp_github_get_me(), mcp_github_create_repository(), mcp_github_create_or_update_file(), mcp_github_push_files()
- **Workflow**: 1) get_me() → 2) create_repository() → 3) create_or_update_file() или push_files() → 4) push_files() для коммитов
- **Ограничения**: push_files() не работает с большими файлами - нужно использовать create_or_update_file() для каждого файла
- **Troubleshooting**: "Git Repository is empty" → Создать файлы с create_or_update_file, "Resource not accessible by integration" → Проверить права доступа, "Not Found" → Проверить правильность owner/repo, "Validation Failed" → Проверить SHA файла

#### 🎯 Использование только для AI Assistant
```
Используйте MCP GitHub для всех операций с репозиторием. 
Создайте файлы с get_me, 
затем используйте create_or_update_file или push_files для коммитов
```

## 🎯 Troubleshooting
- "Git Repository is empty" → Создать файлы с create_or_update_file
- "Resource not accessible by integration" → Проверить права доступа
- "Not Found" → Проверить правильность owner/repo
- "Validation Failed" → Проверить SHA файла

## 🎯 Новые FFT Архитектуры (2025-10-19)

### 🚀 Поддерживаемые платформы
- **Платформы**: NVIDIA, OpenCL, Vulkan, ROCm
- **ROCm поддержка**: Создан для AMD GPU с rocFFT, rocBLAS
- **Производительность**: Сравнение с NVIDIA
- **JSON отчеты**: Автоматизированные результаты тестов

### 🎯 Цели проекта
- **FFT256 > 1.0 TFLOP/s** на всех платформах
- **Нативные реализации** сопоставимы с библиотечными
- **Параллельные вычисления**: 2×FFT512 + 1×FFT1024 в блоке 1024
- **Sliding FFT**: С окнами Хемминга, размеры FFT16 до FFT32768

### 🎯 GitHub MCP Integration

#### 🚀 Настройка для GitHub MCP
- **Аутентификация**: Использовать Personal Access Token (PAT) для доступа к private repo, workflow, write:packages, delete:packages, admin:org, gist, notifications, user, delete_repo
- **Основные функции**: mcp_github_get_me(), mcp_github_create_repository(), mcp_github_create_or_update_file(), mcp_github_push_files()
- **Workflow**: 1) get_me() → 2) create_repository() → 3) create_or_update_file() или push_files() → 4) push_files() для коммитов
- **Ограничения**: push_files() не работает с большими файлами - нужно использовать create_or_update_file() для каждого файла
- **Troubleshooting**: "Git Repository is empty" → Создать файлы с create_or_update_file, "Resource not accessible by integration" → Проверить права доступа, "Not Found" → Проверить правильность owner/repo, "Validation Failed" → Проверить SHA файла

#### 🎯 Использование только для AI Assistant
```
Используйте MCP GitHub для всех операций с репозиторием. 
Создайте файлы с get_me, 
затем используйте create_or_update_file или push_files для коммитов
```

## 🎯 Troubleshooting
- "Git Repository is empty" → Создать файлы с create_or_update_file
- "Resource not accessible by integration" → Проверить права доступа
- "Not Found" → Проверить правильность owner/repo
- "Validation Failed" → Проверить SHA файла

---
*Обновлено: 2025-10-19*  
*Версия: 1.3*
