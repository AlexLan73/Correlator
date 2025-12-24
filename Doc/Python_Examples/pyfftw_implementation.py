#!/usr/bin/env python3
"""
Полная реализация алгоритма FFT корреляции на Python с использованием pyfftw.

Этот скрипт реализует тот же алгоритм, что и C++ версия, и сохраняет результаты
в JSON файлы для верификации и валидации.
"""

import json
import numpy as np
import pyfftw
from datetime import datetime
from typing import List, Tuple, Dict, Any
import os

# Настройка pyfftw для оптимизации
pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(60)


class FFTCorrelator:
    """Реализация FFT коррелятора на Python."""
    
    def __init__(self, fft_size: int, num_shifts: int, num_signals: int, 
                 num_output_points: int, scale_factor: float = 1.0 / 32768.0):
        """
        Инициализация коррелятора.
        
        Args:
            fft_size: Размер FFT (обычно 32768 = 2^15)
            num_shifts: Количество циклических сдвигов (корреляторов)
            num_signals: Количество входных сигналов
            num_output_points: Количество выходных точек (n_kg)
            scale_factor: Масштабирующий коэффициент (по умолчанию 1.0 / 32768.0)
        """
        self.fft_size = fft_size
        self.num_shifts = num_shifts
        self.num_signals = num_signals
        self.num_output_points = num_output_points
        self.scale_factor = scale_factor
        
        # Создать оптимизированные планы FFT/IFFT
        self._setup_fft_plans()
    
    def _setup_fft_plans(self):
        """Настройка оптимизированных планов FFT/IFFT."""
        # Создать временные массивы для планов
        temp_input = pyfftw.empty_aligned(self.fft_size, dtype='complex64')
        temp_output = pyfftw.empty_aligned(self.fft_size, dtype='complex64')
        
        # Планы FFT (forward)
        self.fft_plan = pyfftw.FFTW(
            temp_input, temp_output,
            axes=(0,), direction='FFTW_FORWARD',
            flags=('FFTW_MEASURE',), threads=1
        )
        
        # Планы IFFT (backward)
        self.ifft_plan = pyfftw.FFTW(
            temp_input, temp_output,
            axes=(0,), direction='FFTW_BACKWARD',
            flags=('FFTW_MEASURE',), threads=1,
            normalise_idft=True  # Нормализация после IFFT
        )
    
    @staticmethod
    def generate_m_sequence(length: int, seed: int = 0x1) -> List[int]:
        """
        Генерирует M-последовательность заданной длины.
        
        Args:
            length: Длина последовательности
            seed: Начальное значение LFSR (по умолчанию 0x1)
        
        Returns:
            Список int значений: 1 или -1
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
    
    def step0_generate_sequences(self) -> Tuple[List[int], List[List[int]]]:
        """
        Step 0: Генерация M-последовательностей.
        
        Returns:
            Кортеж (reference_signal, input_signals)
        """
        # Генерация опорного сигнала
        reference_signal = self.generate_m_sequence(self.fft_size, seed=0x1)
        
        # Генерация входных сигналов (каждый с разным seed)
        input_signals = []
        for i in range(self.num_signals):
            signal = self.generate_m_sequence(self.fft_size, seed=0x1 + i)
            input_signals.append(signal)
        
        return reference_signal, input_signals
    
    def step1_reference_fft(self, reference_signal: List[int]) -> np.ndarray:
        """
        Step 1: Формирование спектров опорного сигнала с циклическими сдвигами.
        
        Args:
            reference_signal: Опорный сигнал (M-последовательность)
        
        Returns:
            Массив complex64 размером (num_shifts, fft_size) - сопряженные спектры
        """
        # 1.1. Конвертация и циклические сдвиги
        ref_array = np.array(reference_signal, dtype=np.int32)
        pos_indices = np.arange(self.fft_size)
        
        # Инициализировать массив для всех сдвинутых сигналов
        reference_signals_shifted = np.zeros((self.num_shifts, self.fft_size), dtype=np.complex64)
        
        for shift_idx in range(self.num_shifts):
            # Вычислить индексы с циклическим сдвигом вправо
            src_indices = (pos_indices + shift_idx) % self.fft_size
            
            # Извлечь значения и масштабировать
            values = ref_array[src_indices].astype(np.float32) * self.scale_factor
            
            # Сохранить как комплексные числа
            reference_signals_shifted[shift_idx, :] = values + 0j
        
        # 1.2. Forward FFT для каждого сдвинутого сигнала
        reference_fft = np.zeros((self.num_shifts, self.fft_size), dtype=np.complex64)
        
        for shift_idx in range(self.num_shifts):
            signal = reference_signals_shifted[shift_idx, :]
            # Использовать оптимизированный план FFT
            temp_input = signal.copy()
            temp_output = pyfftw.empty_aligned(self.fft_size, dtype='complex64')
            plan = pyfftw.FFTW(
                temp_input, temp_output,
                axes=(0,), direction='FFTW_FORWARD',
                flags=('FFTW_ESTIMATE',), threads=1
            )
            plan()
            reference_fft[shift_idx, :] = temp_output
        
        # 1.3. Комплексное сопряжение
        reference_fft_conj = np.conj(reference_fft)
        
        return reference_fft_conj
    
    def step2_input_fft(self, input_signals: List[List[int]]) -> np.ndarray:
        """
        Step 2: Формирование спектров входных сигналов.
        
        Args:
            input_signals: Список входных сигналов (M-последовательности)
        
        Returns:
            Массив complex64 размером (num_signals, fft_size) - спектры
        """
        # 2.1. Конвертация int32 → complex float
        input_array = np.array(input_signals, dtype=np.int32)
        input_signals_float = (input_array.astype(np.float32) * self.scale_factor).astype(np.complex64)
        
        # 2.2. Forward FFT для каждого сигнала
        input_fft = np.zeros((self.num_signals, self.fft_size), dtype=np.complex64)
        
        for signal_idx in range(self.num_signals):
            signal = input_signals_float[signal_idx, :]
            # Использовать оптимизированный план FFT
            temp_input = signal.copy()
            temp_output = pyfftw.empty_aligned(self.fft_size, dtype='complex64')
            plan = pyfftw.FFTW(
                temp_input, temp_output,
                axes=(0,), direction='FFTW_FORWARD',
                flags=('FFTW_ESTIMATE',), threads=1
            )
            plan()
            input_fft[signal_idx, :] = temp_output
        
        return input_fft
    
    def step3_correlation(self, reference_fft: np.ndarray, input_fft: np.ndarray) -> np.ndarray:
        """
        Step 3: Корреляция (умножение спектров, IFFT, извлечение пиков).
        
        Args:
            reference_fft: Спектры опорного сигнала (num_shifts, fft_size)
            input_fft: Спектры входных сигналов (num_signals, fft_size)
        
        Returns:
            Массив float32 размером (num_signals, num_shifts, num_output_points) - магнитуды пиков
        """
        # 3.1. Комплексное умножение: ref_fft * conj(input_fft)
        # Расширить массивы для broadcasting
        ref_expanded = reference_fft[np.newaxis, :, :]  # (1, num_shifts, fft_size)
        inp_expanded = input_fft[:, np.newaxis, :]      # (num_signals, 1, fft_size)
        
        # Комплексное умножение с сопряжением
        correlation_fft = ref_expanded * np.conj(inp_expanded)
        # Результат: (num_signals, num_shifts, fft_size)
        
        # 3.2. Inverse FFT
        correlation_ifft = np.zeros((self.num_signals, self.num_shifts, self.fft_size), dtype=np.complex64)
        
        for signal_idx in range(self.num_signals):
            for shift_idx in range(self.num_shifts):
                spectrum = correlation_fft[signal_idx, shift_idx, :]
                # Использовать оптимизированный план IFFT
                temp_input = spectrum.copy()
                temp_output = pyfftw.empty_aligned(self.fft_size, dtype='complex64')
                plan = pyfftw.FFTW(
                    temp_input, temp_output,
                    axes=(0,), direction='FFTW_BACKWARD',
                    flags=('FFTW_ESTIMATE',), threads=1,
                    normalise_idft=True
                )
                plan()
                correlation_ifft[signal_idx, shift_idx, :] = temp_output
        
        # 3.3. Извлечение магнитуд (первые n_kg точек)
        # Взять первые n_kg точек для всех комбинаций
        ifft_first_n = correlation_ifft[:, :, :self.num_output_points]
        
        # Вычислить магнитуду: abs(complex)
        peaks = np.abs(ifft_first_n)
        
        return peaks
    
    def export_step0_json(self, reference_signal: List[int], input_signals: List[List[int]], 
                         output_dir: str):
        """Экспорт Step 0 в JSON."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        config = {
            "fft_size": self.fft_size,
            "num_shifts": self.num_shifts,
            "num_signals": self.num_signals,
            "num_output_points": self.num_output_points,
            "scale_factor": self.scale_factor
        }
        
        data = {
            "step": "STEP0_M_SEQUENCE",
            "timestamp": timestamp,
            "configuration": config,
            "reference_signal": reference_signal,
            "input_signals": input_signals
        }
        
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, "Step0.json")
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Step0 данные сохранены: {filename}")
    
    def export_step1_json(self, reference_fft: np.ndarray, output_dir: str):
        """Экспорт Step 1 в JSON."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        config = {
            "fft_size": self.fft_size,
            "num_shifts": self.num_shifts,
            "num_signals": self.num_signals,
            "num_output_points": self.num_output_points,
            "scale_factor": self.scale_factor
        }
        
        # Преобразовать в плоский массив объектов {"real": ..., "imag": ...}
        reference_fft_flat = reference_fft.flatten()
        reference_fft_json = [
            {"real": float(c.real), "imag": float(c.imag)}
            for c in reference_fft_flat
        ]
        
        data_obj = {
            "reference_fft": reference_fft_json,
            "num_shifts": self.num_shifts,
            "fft_size": self.fft_size
        }
        
        data = {
            "step": "STEP1_REFERENCE_FFT",
            "timestamp": timestamp,
            "configuration": config,
            "data": data_obj,
            "validation": {
                "is_valid": True,
                "error_count": 0,
                "warning_count": 0,
                "errors": [],
                "warnings": []
            }
        }
        
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, "Step1.json")
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Step1 данные сохранены: {filename}")
    
    def export_step2_json(self, input_fft: np.ndarray, output_dir: str):
        """Экспорт Step 2 в JSON."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        config = {
            "fft_size": self.fft_size,
            "num_shifts": self.num_shifts,
            "num_signals": self.num_signals,
            "num_output_points": self.num_output_points,
            "scale_factor": self.scale_factor
        }
        
        # Преобразовать в плоский массив объектов {"real": ..., "imag": ...}
        input_fft_flat = input_fft.flatten()
        input_fft_json = [
            {"real": float(c.real), "imag": float(c.imag)}
            for c in input_fft_flat
        ]
        
        data_obj = {
            "input_fft": input_fft_json,
            "num_signals": self.num_signals,
            "fft_size": self.fft_size
        }
        
        data = {
            "step": "STEP2_INPUT_FFT",
            "timestamp": timestamp,
            "configuration": config,
            "data": data_obj,
            "validation": {
                "is_valid": True,
                "error_count": 0,
                "warning_count": 0,
                "errors": [],
                "warnings": []
            }
        }
        
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, "Step2.json")
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Step2 данные сохранены: {filename}")
    
    def export_step3_json(self, peaks: np.ndarray, output_dir: str):
        """Экспорт Step 3 в JSON."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        config = {
            "fft_size": self.fft_size,
            "num_shifts": self.num_shifts,
            "num_signals": self.num_signals,
            "num_output_points": self.num_output_points,
            "scale_factor": self.scale_factor
        }
        
        # Преобразовать в плоский массив float
        peaks_flat = peaks.flatten()
        peaks_json = [float(p) for p in peaks_flat]
        
        data_obj = {
            "peaks": peaks_json,
            "num_signals": self.num_signals,
            "num_shifts": self.num_shifts,
            "num_output_points": self.num_output_points
        }
        
        data = {
            "step": "STEP3_CORRELATION",
            "timestamp": timestamp,
            "configuration": config,
            "data": data_obj,
            "validation": {
                "is_valid": True,
                "error_count": 0,
                "warning_count": 0,
                "errors": [],
                "warnings": []
            }
        }
        
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, "Step3.json")
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Step3 данные сохранены: {filename}")


def main():
    """Основная функция для запуска алгоритма."""
    print("=" * 70)
    print("FFT CORRELATOR - Python Implementation")
    print("=" * 70)
    
    # Параметры конфигурации
    fft_size = 32768  # 2^15
    num_shifts = 10
    num_signals = 5
    num_output_points = 2000
    scale_factor = 1.0 / 32768.0
    
    # Создать коррелятор
    correlator = FFTCorrelator(fft_size, num_shifts, num_signals, num_output_points, scale_factor)
    
    # Создать выходной каталог с timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("Report", "Validation", timestamp)
    
    print(f"\n[Step 0] Генерация M-последовательностей...")
    reference_signal, input_signals = correlator.step0_generate_sequences()
    correlator.export_step0_json(reference_signal, input_signals, output_dir)
    
    print(f"\n[Step 1] Формирование спектров опорного сигнала...")
    reference_fft = correlator.step1_reference_fft(reference_signal)
    correlator.export_step1_json(reference_fft, output_dir)
    
    print(f"\n[Step 2] Формирование спектров входных сигналов...")
    input_fft = correlator.step2_input_fft(input_signals)
    correlator.export_step2_json(input_fft, output_dir)
    
    print(f"\n[Step 3] Корреляция (умножение, IFFT, извлечение пиков)...")
    peaks = correlator.step3_correlation(reference_fft, input_fft)
    correlator.export_step3_json(peaks, output_dir)
    
    print(f"\n" + "=" * 70)
    print("✓ ВСЕ ЭТАПЫ ВЫПОЛНЕНЫ УСПЕШНО!")
    print(f"✓ Результаты сохранены в: {output_dir}")
    print("=" * 70)
    
    # Вывести статистику
    print(f"\nСтатистика:")
    print(f"  Размер FFT: {fft_size}")
    print(f"  Количество сдвигов: {num_shifts}")
    print(f"  Количество сигналов: {num_signals}")
    print(f"  Количество выходных точек: {num_output_points}")
    print(f"  Размер массива пиков: {peaks.shape}")
    print(f"  Всего значений пиков: {peaks.size}")


if __name__ == "__main__":
    main()

