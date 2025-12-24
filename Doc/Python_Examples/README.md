# Python реализация FFT коррелятора

Этот каталог содержит реализацию алгоритма FFT корреляции на Python для верификации и валидации C++ реализации.

## Установка зависимостей

### Вариант 1: Использование pyfftw (рекомендуется)

```bash
pip install -r requirements.txt
```

**Требования:**
- Python 3.7+
- NumPy
- pyfftw (требует установленного FFTW библиотеки)
  - Linux: `sudo apt-get install libfftw3-dev` (Debian/Ubuntu)
  - macOS: `brew install fftw`
  - Windows: Скачать с официального сайта FFTW

### Вариант 2: Использование PyTorch (альтернатива)

Если pyfftw недоступен, можно использовать PyTorch версию:

```bash
pip install numpy torch
```

PyTorch версия может быть быстрее на GPU (если доступен CUDA).

## Использование

### Запуск pyfftw версии

```bash
python pyfftw_implementation.py
```

### Запуск PyTorch версии

```bash
python pytorch_implementation.py
```

## Выходные файлы

Результаты сохраняются в каталог `Report/Validation/YYYY-MM-DD_HH-MM-SS/`:

- `Step0.json` - M-последовательности (reference_signal и input_signals)
- `Step1.json` - Спектры опорного сигнала с циклическими сдвигами
- `Step2.json` - Спектры входных сигналов
- `Step3.json` - Результаты корреляции (магнитуды пиков)

## Сравнение с C++ реализацией

Для верификации результатов сравните JSON файлы Python версии с соответствующими файлами C++ версии.

**Важно:** 
- Формат JSON должен точно совпадать
- Допускается погрешность округления до `1e-5` для float32 значений
- Все массивы должны иметь одинаковый порядок элементов

## Параметры конфигурации

По умолчанию используются следующие параметры (совпадают с C++ версией):

```python
fft_size = 32768           # Размер FFT (2^15)
num_shifts = 10            # Количество циклических сдвигов
num_signals = 5            # Количество входных сигналов
num_output_points = 2000   # Количество выходных точек
scale_factor = 1.0 / 32768.0  # Масштабирующий коэффициент
```

Для изменения параметров отредактируйте значения в функции `main()` в соответствующих файлах.

## Структура данных

### Step 0 (Step0.json)
- `reference_signal`: Плоский массив int32 значений
- `input_signals`: Массив массивов (num_signals сигналов по fft_size элементов)

### Step 1 (Step1.json)
- `data.reference_fft`: Плоский массив объектов `{"real": float, "imag": float}`
  - Порядок: `[shift_0_pos_0, shift_0_pos_1, ..., shift_0_pos_N-1, shift_1_pos_0, ...]`

### Step 2 (Step2.json)
- `data.input_fft`: Плоский массив объектов `{"real": float, "imag": float}`
  - Порядок: `[signal_0_pos_0, signal_0_pos_1, ..., signal_0_pos_N-1, signal_1_pos_0, ...]`

### Step 3 (Step3.json)
- `data.peaks`: Плоский массив float значений
  - Порядок: `[signal_0_shift_0_point_0, signal_0_shift_0_point_1, ..., signal_0_shift_0_point_n_kg-1, signal_0_shift_1_point_0, ...]`

## Алгоритм

Подробное описание алгоритма см. в `Doc/Python_Algorithm_Description.md`.

## Примечания

1. **Производительность:** pyfftw версия обычно быстрее на CPU, PyTorch версия может использовать GPU.

2. **Точность:** Обе версии используют float32 (complex64) для совместимости с C++ реализацией.

3. **Нормализация IFFT:** 
   - pyfftw: Использует параметр `normalise_idft=True` для автоматической нормализации
   - PyTorch: `torch.fft.ifft` нормализует автоматически
   - При сравнении с C++ убедитесь, что нормализация совпадает

4. **Масштабирование:** Все значения масштабируются из диапазона `[-32768, 32767]` в `[-1.0, 1.0]` с помощью `scale_factor = 1.0 / 32768.0`.

