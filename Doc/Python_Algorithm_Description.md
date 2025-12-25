# Описание алгоритма FFT корреляции для Python реализации

## Цель

Этот документ описывает алгоритм FFT корреляции для реализации на Python с использованием `pyfftw` или `PyTorch`. Алгоритм предназначен для верификации и валидации C++ реализации.

## Общая структура алгоритма

Алгоритм состоит из 4 шагов:

1. **Step 0**: Генерация M-последовательностей
2. **Step 1**: Формирование спектров опорного сигнала с циклическими сдвигами
3. **Step 2**: Формирование спектров тестовых сигналов
4. **Step 3**: Корреляция (умножение спектров, IFFT, извлечение пиков)

---

## Конфигурация

Параметры алгоритма к примеру такие  (должны совпадать с C++ реализацией):

```python
config = {
    "fft_size": 32768,           # Размер FFT (2^15)
    "num_shifts": 10,            # Количество циклических сдвигов (корреляторов)
    "num_signals": 5,            # Количество входных сигналов
    "num_output_points": 2000,   # Количество выходных точек (n_kg)
    "scale_factor": 1.0 / 32768.0  # Масштабирующий коэффициент
}
```

---

## Step 0: Генерация M-последовательности

### Алгоритм

M-последовательность генерируется с помощью Linear Feedback Shift Register (LFSR) с полиномом `0xB8000000`.

### Реализация на Python

```python
def generate_m_sequence(length, seed=0x1):
    """
    Генерирует M-последовательность заданной длины.
    
    Args:
        length: Длина последовательности
        seed: Начальное значение LFSR (по умолчанию 0x1)
    
    Returns:
        Список int32 значений: 1 или -1
    """
    sequence = []
    lfsr = seed
    POLY = 0xB8000000
    
    for i in range(length):
        # Извлечь старший бит (bit 31)
        bit = (lfsr >> 31) & 1
        
        # Записать значение: 1 для bit=1, -1 для bit=0
        sequence.append(1 if bit else -1)
        
        # Обновить LFSR
        if bit:
            lfsr = ((lfsr << 1) ^ POLY) & 0xFFFFFFFF
        else:
            lfsr = (lfsr << 1) & 0xFFFFFFFF
    
    return sequence
```

### Формат JSON (Step0.json)

```json
{
  "step": "STEP0_M_SEQUENCE",
  "timestamp": "2025-12-24 10:00:00",
  "configuration": {
    "fft_size": 32768,
    "num_shifts": 10,
    "num_signals": 5,
    "num_output_points": 2000,
    "scale_factor": 0.000030518
  },
  "reference_signal": [1, -1, 1, ...],
  "input_signals": [
    [1, -1, 1, ...],  # Сигнал 0
    [1, -1, 1, ...],  # Сигнал 1
    ...                # Сигналы 2-4
  ]
}
```

---

## Step 1: Формирование спектров опорного сигнала

### Описание

На этом шаге:
1. Опорный сигнал (M-последовательность) конвертируется из `int32` в `complex float`
2. Применяются циклические сдвиги (все `num_shifts` вариантов)
3. Выполняется Forward FFT для каждого сдвинутого сигнала
4. Применяется комплексное сопряжение к результатам FFT

### 1.1. Pre-callback: Конвертация int32 → complex float

#### Функция: `prepare_reference_signals_pre_callback`

**Входные данные:**
- `reference_signal`: `list[int32]` размером `fft_size`

**Выходные данные:**
- `reference_signals_shifted`: `numpy.ndarray[complex64]` размером `(num_shifts, fft_size)`

**Алгоритм:**

```python
import numpy as np

def prepare_reference_signals_pre_callback(reference_signal, num_shifts, fft_size, scale_factor):
    """
    Конвертирует опорный сигнал и применяет циклические сдвиги.
    
    Args:
        reference_signal: Список int32 значений (M-последовательность)
        num_shifts: Количество циклических сдвигов
        fft_size: Размер FFT
        scale_factor: Масштабирующий коэффициент (обычно 1.0 / 32768.0)
    
    Returns:
        Массив complex64 размером (num_shifts, fft_size)
    """
    # Инициализировать выходной массив
    output = np.zeros((num_shifts, fft_size), dtype=np.complex64)
    
    # Преобразовать в numpy массив
    ref_array = np.array(reference_signal, dtype=np.int32)
    
    # Для каждого сдвига
    for shift_idx in range(num_shifts):
        # Для каждой позиции
        for pos_idx in range(fft_size):
            # Вычислить исходный индекс с циклическим сдвигом вправо
            # Формула: src_idx = (pos_idx + shift) % fft_size
            src_idx = (pos_idx + shift_idx) % fft_size
            
            # Прочитать значение из исходного сигнала
            sample_int32 = ref_array[src_idx]
            
            # Конвертировать в float и масштабировать
            real_part = float(sample_int32) * scale_factor
            imag_part = 0.0  # Вещественный сигнал
            
            # Сохранить в выходной массив
            output[shift_idx, pos_idx] = complex(real_part, imag_part)
    
    return output
```

**Оптимизированная версия (векторизация):**

```python
def prepare_reference_signals_vectorized(reference_signal, num_shifts, fft_size, scale_factor):
    """
    Векторизованная версия с использованием numpy.
    """
    ref_array = np.array(reference_signal, dtype=np.int32)
    
    # Создать массив индексов для всех позиций
    pos_indices = np.arange(fft_size)
    
    # Инициализировать выходной массив
    output = np.zeros((num_shifts, fft_size), dtype=np.complex64)
    
    for shift_idx in range(num_shifts):
        # Вычислить индексы с циклическим сдвигом
        src_indices = (pos_indices + shift_idx) % fft_size
        
        # Извлечь значения и масштабировать
        values = ref_array[src_indices].astype(np.float32) * scale_factor
        
        # Сохранить как комплексные числа
        output[shift_idx, :] = values + 0j
    
    return output
```

### 1.2. Forward FFT

**Алгоритм:**

```python
import pyfftw

def compute_reference_fft(reference_signals_shifted, fft_size):
    """
    Выполняет Forward FFT для каждого сдвинутого сигнала.
    
    Args:
        reference_signals_shifted: Массив complex64 размером (num_shifts, fft_size)
        fft_size: Размер FFT
    
    Returns:
        Массив complex64 размером (num_shifts, fft_size) - спектры
    """
    num_shifts = reference_signals_shifted.shape[0]
    reference_fft = np.zeros((num_shifts, fft_size), dtype=np.complex64)
    
    for shift_idx in range(num_shifts):
        # Выполнить FFT
        signal = reference_signals_shifted[shift_idx, :]
        reference_fft[shift_idx, :] = pyfftw.interfaces.numpy_fft.fft(signal, fft_size)
    
    return reference_fft
```

### 1.3. Post-callback: Комплексное сопряжение

#### Функция: `post_callback_conjugate`

**Алгоритм:**

Комплексное сопряжение: `(real, imag) → (real, -imag)`

**Реализация:**

```python
def apply_complex_conjugate(reference_fft):
    """
    Применяет комплексное сопряжение к спектрам.
    
    Args:
        reference_fft: Массив complex64 размером (num_shifts, fft_size)
    
    Returns:
        Массив complex64 с сопряженными значениями
    """
    # Комплексное сопряжение: conj(a + bi) = a - bi
    return np.conj(reference_fft)
```

**Эквивалентно:**

```python
# Вручную:
reference_fft_conj = np.zeros_like(reference_fft)
reference_fft_conj.real = reference_fft.real
reference_fft_conj.imag = -reference_fft.imag
```

### Формат JSON (Step1.json)

```json
{
  "step": "STEP1_REFERENCE_FFT",
  "timestamp": "2025-12-24 10:00:01",
  "configuration": { ... },
  "data": {
    "reference_fft": [
      {"real": 0.123456, "imag": -0.789012},  # shift 0, pos 0
      {"real": 0.234567, "imag": -0.890123},  # shift 0, pos 1
      ...                                      # Всего num_shifts * fft_size элементов
    ],
    "num_shifts": 10,
    "fft_size": 32768
  },
  "validation": { ... }
}
```

**Примечание:** Данные в JSON представлены как плоский массив. Порядок: `[shift_0_pos_0, shift_0_pos_1, ..., shift_0_pos_N-1, shift_1_pos_0, ...]`

---

## Step 2: Формирование спектров тестовых сигналов

### Описание

На этом шаге:
1. Входные сигналы (M-последовательности) конвертируются из `int32` в `complex float`
2. Выполняется Forward FFT для каждого входного сигнала

### 2.1. Pre-callback: Конвертация int32 → complex float

#### Функция: `prepare_input_signals_pre_callback`

**Входные данные:**
- `input_signals`: `list[list[int32]]` размером `(num_signals, fft_size)`

**Выходные данные:**
- `input_signals_float`: `numpy.ndarray[complex64]` размером `(num_signals, fft_size)`

**Алгоритм:**

```python
def prepare_input_signals_pre_callback(input_signals, num_signals, fft_size, scale_factor):
    """
    Конвертирует входные сигналы из int32 в complex float.
    
    Args:
        input_signals: Список списков int32 (num_signals сигналов по fft_size элементов)
        num_signals: Количество входных сигналов
        fft_size: Размер каждого сигнала
        scale_factor: Масштабирующий коэффициент
    
    Returns:
        Массив complex64 размером (num_signals, fft_size)
    """
    # Преобразовать в numpy массив
    input_array = np.array(input_signals, dtype=np.int32)
    
    # Конвертировать и масштабировать
    output = (input_array.astype(np.float32) * scale_factor).astype(np.complex64)
    
    return output
```

### 2.2. Forward FFT

**Алгоритм:**

```python
def compute_input_fft(input_signals_float, fft_size):
    """
    Выполняет Forward FFT для каждого входного сигнала.
    
    Args:
        input_signals_float: Массив complex64 размером (num_signals, fft_size)
        fft_size: Размер FFT
    
    Returns:
        Массив complex64 размером (num_signals, fft_size) - спектры
    """
    num_signals = input_signals_float.shape[0]
    input_fft = np.zeros((num_signals, fft_size), dtype=np.complex64)
    
    for signal_idx in range(num_signals):
        signal = input_signals_float[signal_idx, :]
        input_fft[signal_idx, :] = pyfftw.interfaces.numpy_fft.fft(signal, fft_size)
    
    return input_fft
```

### Формат JSON (Step2.json)

```json
{
  "step": "STEP2_INPUT_FFT",
  "timestamp": "2025-12-24 10:00:02",
  "configuration": { ... },
  "data": {
    "input_fft": [
      {"real": 0.345678, "imag": 0.901234},  # signal 0, pos 0
      {"real": 0.456789, "imag": 0.012345},  # signal 0, pos 1
      ...                                    # Всего num_signals * fft_size элементов
    ],
    "num_signals": 5,
    "fft_size": 32768
  },
  "validation": { ... }
}
```

---

## Step 3: Корреляция (умножение спектров, IFFT, извлечение пиков)

### Описание

На этом шаге:
1. Выполняется комплексное умножение: `correlation_fft = reference_fft * conj(input_fft)`
2. Выполняется Inverse FFT для каждой комбинации сигнала и сдвига
3. Извлекаются первые `n_kg` точек и вычисляется их магнитуда

### 3.1. Pre-callback: Комплексное умножение

#### Функция: `complex_multiply_kernel`

**Алгоритм:**

Корреляция в частотной области: `result = ref_fft * conj(input_fft)`

**Математика комплексного умножения:**

Если `ref_fft = a + bi` и `input_fft = c + di`, то:
- `conj(input_fft) = c - di`
- `result = (a + bi) * (c - di) = (ac + bd) + (bc - ad)i`

**Реализация:**

```python
def compute_correlation_fft(reference_fft, input_fft, num_signals, num_shifts, fft_size):
    """
    Выполняет комплексное умножение для корреляции в частотной области.
    
    Args:
        reference_fft: Массив complex64 размером (num_shifts, fft_size)
        input_fft: Массив complex64 размером (num_signals, fft_size)
        num_signals: Количество входных сигналов
        num_shifts: Количество сдвигов
        fft_size: Размер FFT
    
    Returns:
        Массив complex64 размером (num_signals, num_shifts, fft_size)
    """
    # Инициализировать выходной массив
    correlation_fft = np.zeros((num_signals, num_shifts, fft_size), dtype=np.complex64)
    
    # Для каждой комбинации сигнала и сдвига
    for signal_idx in range(num_signals):
        for shift_idx in range(num_shifts):
            # ref_fft[shift_idx, :] - спектр опорного сигнала для данного сдвига
            # input_fft[signal_idx, :] - спектр входного сигнала
            # Выполнить: ref_fft * conj(input_fft)
            ref_spectrum = reference_fft[shift_idx, :]
            inp_spectrum = input_fft[signal_idx, :]
            
            # Комплексное умножение с сопряжением
            correlation_fft[signal_idx, shift_idx, :] = ref_spectrum * np.conj(inp_spectrum)
    
    return correlation_fft
```

**Векторизованная версия:**

```python
def compute_correlation_fft_vectorized(reference_fft, input_fft, num_signals, num_shifts, fft_size):
    """
    Векторизованная версия комплексного умножения.
    """
    # Расширить массивы для broadcasting
    # reference_fft: (num_shifts, fft_size) → (1, num_shifts, fft_size)
    # input_fft: (num_signals, fft_size) → (num_signals, 1, fft_size)
    ref_expanded = reference_fft[np.newaxis, :, :]  # (1, num_shifts, fft_size)
    inp_expanded = input_fft[:, np.newaxis, :]      # (num_signals, 1, fft_size)
    
    # Комплексное умножение с сопряжением
    correlation_fft = ref_expanded * np.conj(inp_expanded)
    
    # Результат: (num_signals, num_shifts, fft_size)
    return correlation_fft
```

### 3.2. Inverse FFT

**Алгоритм:**

```python
def compute_correlation_ifft(correlation_fft, fft_size):
    """
    Выполняет Inverse FFT для каждой комбинации сигнала и сдвига.
    
    Args:
        correlation_fft: Массив complex64 размером (num_signals, num_shifts, fft_size)
        fft_size: Размер FFT
    
    Returns:
        Массив complex64 размером (num_signals, num_shifts, fft_size) - временная область
    """
    num_signals, num_shifts, _ = correlation_fft.shape
    correlation_ifft = np.zeros((num_signals, num_shifts, fft_size), dtype=np.complex64)
    
    for signal_idx in range(num_signals):
        for shift_idx in range(num_shifts):
            spectrum = correlation_fft[signal_idx, shift_idx, :]
            correlation_ifft[signal_idx, shift_idx, :] = pyfftw.interfaces.numpy_fft.ifft(spectrum, fft_size)
    
    return correlation_ifft
```

### 3.3. Post-callback: Извлечение магнитуд

#### Функция: `post_callback_extract_magnitudes`

**Важно:** Этот callback **НЕ ищет максимум** в диапазоне. Он просто берет первые `n_kg` точек из IFFT результата и вычисляет их магнитуду.

**Алгоритм:**

```python
def extract_peaks_magnitudes(correlation_ifft, num_signals, num_shifts, n_kg):
    """
    Извлекает первые n_kg точек из IFFT результата и вычисляет их магнитуду.
    
    Args:
        correlation_ifft: Массив complex64 размером (num_signals, num_shifts, fft_size)
        num_signals: Количество входных сигналов
        num_shifts: Количество сдвигов
        n_kg: Количество выходных точек (num_output_points)
    
    Returns:
        Массив float32 размером (num_signals, num_shifts, n_kg) - магнитуды
    """
    # Инициализировать выходной массив
    peaks = np.zeros((num_signals, num_shifts, n_kg), dtype=np.float32)
    
    # Для каждой комбинации сигнала и сдвига
    for signal_idx in range(num_signals):
        for shift_idx in range(num_shifts):
            # Взять первые n_kg точек из IFFT результата
            ifft_result = correlation_ifft[signal_idx, shift_idx, :n_kg]
            
            # Вычислить магнитуду: magnitude = sqrt(real² + imag²) = abs(complex)
            peaks[signal_idx, shift_idx, :] = np.abs(ifft_result)
    
    return peaks
```

**Векторизованная версия:**

```python
def extract_peaks_magnitudes_vectorized(correlation_ifft, num_signals, num_shifts, n_kg):
    """
    Векторизованная версия извлечения магнитуд.
    """
    # Взять первые n_kg точек для всех комбинаций
    ifft_first_n = correlation_ifft[:, :, :n_kg]
    
    # Вычислить магнитуду
    peaks = np.abs(ifft_first_n)
    
    return peaks
```

### Формат JSON (Step3.json)

```json
{
  "step": "STEP3_CORRELATION",
  "timestamp": "2025-12-24 10:00:03",
  "configuration": { ... },
  "data": {
    "peaks": [
      0.123456, 0.234567, 0.345678, ...,  # signal 0, shift 0, n_kg значений
      0.456789, 0.567890, 0.678901, ...,  # signal 0, shift 1, n_kg значений
      ...                                  # Всего num_signals * num_shifts * n_kg элементов
    ],
    "num_signals": 5,
    "num_shifts": 10,
    "num_output_points": 2000
  },
  "validation": { ... }
}
```

**Примечание:** Данные в JSON представлены как плоский массив. Порядок: `[signal_0_shift_0_point_0, signal_0_shift_0_point_1, ..., signal_0_shift_0_point_n_kg-1, signal_0_shift_1_point_0, ...]`

---

## Полный пример использования

См. файл `Doc/Python_Examples/pyfftw_implementation.py` для полной реализации алгоритма.

---

## Примечания

1. **Точность:** Используйте `float32` (или `complex64`) для совместимости с C++ реализацией, которая использует OpenCL `float` и `float2`.

2. **Масштабирование:** `scale_factor` обычно равен `1.0 / 32768.0` для нормализации значений из диапазона `[-32768, 32767]` в `[-1.0, 1.0]`.

3. **FFT нормализация:** PyFFTW и NumPy FFT используют разные нормализации по умолчанию:
   - NumPy: `ifft(fft(x)) = x` (без нормализации)
   - PyFFTW: Можно настроить через параметры
   - C++ clFFT: Обычно требует нормализации вручную после IFFT

4. **Порядок данных в JSON:** Все массивы сохраняются как плоские списки. Необходимо правильно вычислять индексы при чтении/записи.

5. **Валидация:** Сравнение результатов Python и C++ реализаций должно учитывать погрешности округления (обычно допускается разница до `1e-5` для float32).

