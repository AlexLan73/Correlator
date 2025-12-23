# 🎯 ЗАЩИТА ОТ ДВОЙНОГО CLEANUP - БЫСТРЫЙ ГАЙД

## 📍 ГДЕ ДОБАВИТЬ ФЛАГ?

### 1️⃣ **В HEADER (fft_handler.hpp):**

```cpp
struct FFTContext {
    cl_context context;
    cl_command_queue queue;
    cl_device_id device;
    
    // ... остальные поля ...
    
    bool initialized;
    bool is_cleaned_up;  // ← НОВЫЙ ФЛАГ!
    
    FFTContext() 
        : context(nullptr), queue(nullptr), device(nullptr),
          // ... инициализация остальных полей ...
          initialized(false), 
          is_cleaned_up(false)  // ← ИНИЦИАЛИЗИРУЕМ В FALSE
    {}
};
```

---

## 📍 КАК ЗАМЕНИТЬ cleanup()?

### ❌ СТАРЫЙ КОД (тупит):

```cpp
void FFTHandler::cleanup() {
    if(!ctx_.initialized)
        return;
    printf("[FFT] Cleaning up...\n");
    
    // Сразу начинаем освобождение
    if (ctx_.reference_data) clReleaseMemObject(ctx_.reference_data);
    // ... остальное ...
}
```

**Проблема:** При втором вызове падает на `clReleaseMemObject()` нулевого указателя!

---

### ✅ НОВЫЙ КОД (безопасен):

```cpp
void FFTHandler::cleanup() {
    // ✅ ЗАЩИТА 1: Если уже вычищено - не трогаем!
    if (ctx_.is_cleaned_up) {
        printf("[FFT] Already cleaned up, skipping...\n");
        return;  // ← ВЫХОД ЗДЕСЬ!
    }
    
    // ✅ ЗАЩИТА 2: Если не инициализировано - не трогаем!
    if (!ctx_.initialized) {
        printf("[FFT] Not initialized, skipping cleanup\n");
        return;  // ← ВЫХОД ЗДЕСЬ!
    }
    
    printf("[FFT] Cleaning up GPU resources...\n");
    
    // 1. Разрушаем планы ПЕРВЫМИ
    if (ctx_.reference_fft_plan) {
        clfftDestroyPlan(&ctx_.reference_fft_plan);
        ctx_.reference_fft_plan = nullptr;
    }
    // ... остальные планы ...
    
    // 2. Освобождаем память
    if (ctx_.reference_data) {
        clReleaseMemObject(ctx_.reference_data);
        ctx_.reference_data = nullptr;
    }
    // ... остальные буферы ...
    
    // 3. ВАЖНО: Ставим флаги
    ctx_.initialized = false;
    ctx_.is_cleaned_up = true;  // ← СТАВИМ ФЛАГ!
    
    printf("[OK] GPU cleanup complete!\n\n");
}
```

---

## 🎯 ГДЕ ИНИЦИАЛИЗИРОВАТЬ ФЛАГ?

### Вариант 1️⃣: **В конструкторе FFTContext** (РЕКОМЕНДУЕТСЯ)

```cpp
// В fft_handler.hpp

struct FFTContext {
    // ... поля ...
    bool initialized;
    bool is_cleaned_up;
    
    // Конструктор (ВАЖНО!)
    FFTContext() 
        : context(nullptr), 
          queue(nullptr), 
          device(nullptr),
          // ... другие поля инициализируются ...
          initialized(false), 
          is_cleaned_up(false)  // ← ТУТ!
    {}
};
```

### Вариант 2️⃣: **В конструкторе FFTHandler**

```cpp
// В fft_handler.cpp

FFTHandler::FFTHandler(cl_context ctx, cl_command_queue q, cl_device_id dev) {
    ctx_.context = ctx;
    ctx_.queue = q;
    ctx_.device = dev;
    ctx_.initialized = false;
    ctx_.is_cleaned_up = false;  // ← ТУТ!
    
    // ... остальная инициализация ...
}
```

### Вариант 3️⃣: **Где угодно перед первым использованием**

```cpp
// Где угодно в коде, ДО ПЕРВОГО вызова cleanup()

FFTHandler fft_handler(context, queue, device);
// fft_handler.ctx_.is_cleaned_up = false;  // ← можно даже вот так
```

---

## 📊 ЖИЗНЕННЫЙ ЦИКЛ ФЛАГА

```
┌─────────────────────────────────┐
│  FFTHandler создан              │
│  ↓                              │
│  is_cleaned_up = false  ← ТУТ!  │
│  initialized = false            │
└─────────────────────────────────┘
         ↓
┌─────────────────────────────────┐
│  initialize() вызван            │
│  ↓                              │
│  initialized = true             │
│  is_cleaned_up = false          │
└─────────────────────────────────┘
         ↓
┌─────────────────────────────────┐
│  Работа (step1, step2, step3)   │
│  ↓                              │
│  initialized = true             │
│  is_cleaned_up = false          │
└─────────────────────────────────┘
         ↓
┌─────────────────────────────────┐
│  cleanup() вызван первый раз    │
│  ↓                              │
│  Освобождаем ресурсы            │
│  ↓                              │
│  initialized = false            │
│  is_cleaned_up = true  ← ТУТ!   │
└─────────────────────────────────┘
         ↓
┌─────────────────────────────────┐
│  cleanup() вызван второй раз    │
│  ↓                              │
│  if (is_cleaned_up) return;     │ ← ВЫХОД!
│  ↓                              │
│  Ничего не делаем!              │
│  ↓                              │
│  ✅ БЕЗ ОШИБОК!                 │
└─────────────────────────────────┘
```

---

## ✅ ПРОВЕРКА

Когда запустишь, должно быть:

```
[GPU] Initializing OpenCL context...
[GPU] Initializing clFFT library...
[GPU] Creating FFT handler...

[STEP 1] Processing reference signals...
[OK] Step 1 completed!

[STEP 2] Processing input signals...
[OK] Step 2 completed!

[STEP 3] Computing correlation...
[OK] Step 3 completed!

[GPU] Cleaning up...
  1. Destroying FFT plans...
     ✓ Reference FFT plan destroyed
     ✓ Input FFT plan destroyed
     ✓ Correlation IFFT plan destroyed
  2. Releasing GPU memory buffers...
     ✓ Reference data buffer released
     ✓ Reference FFT buffer released
     ✓ Input data buffer released
     ✓ Input FFT buffer released
     ✓ Correlation FFT buffer released
     ✓ Correlation IFFT buffer released
     ✓ Pre-callback userdata buffer released
     ✓ Post-callback userdata buffer released
[OK] GPU cleanup complete!

✅ ШАГ 1, 2 & 3 COMPLETE!
✨ FFT CORRELATOR PIPELINE COMPLETE!
```

**БЕЗ ОШИБОК! БЕЗ ДВОЙНОГО CLEANUP!** 🎉

---

## 📋 РЕЗЮМЕ

| Что | Где | Значение |
|-----|-----|----------|
| **Объявление флага** | `struct FFTContext` | `bool is_cleaned_up;` |
| **Инициализация** | конструктор FFTContext | `is_cleaned_up(false)` |
| **Проверка в cleanup()** | начало функции | `if (ctx_.is_cleaned_up) return;` |
| **Установка флага** | конец cleanup() | `ctx_.is_cleaned_up = true;` |

---

## 🚀 ГОТОВО!

Теперь cleanup() можно вызывать сколько угодно раз - он не сломается! ✅
