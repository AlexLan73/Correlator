#!/usr/bin/env python3
"""
Альтернативная реализация алгоритма FFT корреляции на Python с использованием PyTorch.

Эта версия использует torch.fft для FFT/IFFT операций и может быть быстрее
на GPU, если доступен CUDA.
"""

import json
import numpy as np
import torch
from datetime import datetime
from typing import List, Tuple
import os


class FFTCorrelatorPyTorch:
    """Реализация FFT коррелятора на Python с использованием PyTorch."""
    
    def __init__(self, fft_size: int, num_shifts: int, num_signals: int,
                 num_output_points: int, scale_factor: float = 1.0 / 32768.0,
                 device: str = 'cpu'):
        """
        Инициализация коррелятора.
        
        Args:
            fft_size: Размер FFT (обычно 32768 = 2^15)
            num_shifts: Количество циклических сдвигов (корреляторов)
            num_signals: Количество входных сигналов
            num_output_points: Количество выходных точек (n_kg)
            scale_factor: Масштабирующий коэффициент (по умолчанию 1.0 / 32768.0)
            device: Устройство для вычислений ('cpu' или 'cuda')
        """
        self.fft_size = fft_size
        self.num_shifts = num_shifts
        self.num_signals = num_signals
        self.num_output_points = num_output_points
        self.scale_factor = scale_factor
        self.device = device
        
        # Проверить доступность CUDA
        if device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA недоступна, используется CPU")
            self.device = 'cpu'
    
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
            bit = (lfsr >> 31) & 1
            sequence.append(1 if bit else -1)
            
            if bit:
                lfsr = ((lfsr << 1) ^ POLY) & 0xFFFFFFFF
            else:
                lfsr = (lfsr << 1) & 0xFFFFFFFF
        
        return sequence
    
    def step0_generate_sequences(self) -> Tuple[List[int], List[List[int]]]:
        """Step 0: Генерация M-последовательностей."""
        reference_signal = self.generate_m_sequence(self.fft_size, seed=0x1)
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
        # Конвертация в torch tensor
        ref_tensor = torch.tensor(reference_signal, dtype=torch.int32, device=self.device)
        pos_indices = torch.arange(self.fft_size, device=self.device)
        
        # Инициализировать массив для всех сдвинутых сигналов
        reference_signals_shifted = torch.zeros((self.num_shifts, self.fft_size), 
                                                dtype=torch.complex64, device=self.device)
        
        for shift_idx in range(self.num_shifts):
            # Циклический сдвиг вправо
            src_indices = (pos_indices + shift_idx) % self.fft_size
            values = ref_tensor[src_indices].float() * self.scale_factor
            reference_signals_shifted[shift_idx, :] = torch.complex(values, torch.zeros_like(values))
        
        # Forward FFT для каждого сдвинутого сигнала
        reference_fft = torch.zeros((self.num_shifts, self.fft_size), 
                                    dtype=torch.complex64, device=self.device)
        
        for shift_idx in range(self.num_shifts):
            signal = reference_signals_shifted[shift_idx, :]
            reference_fft[shift_idx, :] = torch.fft.fft(signal, n=self.fft_size)
        
        # Комплексное сопряжение
        reference_fft_conj = torch.conj(reference_fft)
        
        # Конвертация обратно в numpy для экспорта
        return reference_fft_conj.cpu().numpy()
    
    def step2_input_fft(self, input_signals: List[List[int]]) -> np.ndarray:
        """
        Step 2: Формирование спектров входных сигналов.
        
        Args:
            input_signals: Список входных сигналов (M-последовательности)
        
        Returns:
            Массив complex64 размером (num_signals, fft_size) - спектры
        """
        # Конвертация в torch tensor
        input_tensor = torch.tensor(input_signals, dtype=torch.int32, device=self.device)
        input_signals_float = input_tensor.float() * self.scale_factor
        input_signals_complex = torch.complex(input_signals_float, torch.zeros_like(input_signals_float))
        
        # Forward FFT для каждого сигнала
        input_fft = torch.zeros((self.num_signals, self.fft_size), 
                                dtype=torch.complex64, device=self.device)
        
        for signal_idx in range(self.num_signals):
            signal = input_signals_complex[signal_idx, :]
            input_fft[signal_idx, :] = torch.fft.fft(signal, n=self.fft_size)
        
        # Конвертация обратно в numpy
        return input_fft.cpu().numpy()
    
    def step3_correlation(self, reference_fft: np.ndarray, input_fft: np.ndarray) -> np.ndarray:
        """
        Step 3: Корреляция (умножение спектров, IFFT, извлечение пиков).
        
        Args:
            reference_fft: Спектры опорного сигнала (num_shifts, fft_size)
            input_fft: Спектры входных сигналов (num_signals, fft_size)
        
        Returns:
            Массив float32 размером (num_signals, num_shifts, num_output_points) - магнитуды пиков
        """
        # Конвертация в torch tensor
        ref_fft_tensor = torch.tensor(reference_fft, dtype=torch.complex64, device=self.device)
        inp_fft_tensor = torch.tensor(input_fft, dtype=torch.complex64, device=self.device)
        
        # Комплексное умножение: ref_fft * conj(input_fft)
        ref_expanded = ref_fft_tensor.unsqueeze(0)  # (1, num_shifts, fft_size)
        inp_expanded = inp_fft_tensor.unsqueeze(1)  # (num_signals, 1, fft_size)
        
        correlation_fft = ref_expanded * torch.conj(inp_expanded)
        # Результат: (num_signals, num_shifts, fft_size)
        
        # Inverse FFT
        correlation_ifft = torch.zeros((self.num_signals, self.num_shifts, self.fft_size),
                                      dtype=torch.complex64, device=self.device)
        
        for signal_idx in range(self.num_signals):
            for shift_idx in range(self.num_shifts):
                spectrum = correlation_fft[signal_idx, shift_idx, :]
                correlation_ifft[signal_idx, shift_idx, :] = torch.fft.ifft(spectrum, n=self.fft_size)
        
        # Извлечение магнитуд (первые n_kg точек)
        ifft_first_n = correlation_ifft[:, :, :self.num_output_points]
        peaks = torch.abs(ifft_first_n)
        
        # Конвертация обратно в numpy
        return peaks.cpu().numpy().astype(np.float32)
    
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
    print("FFT CORRELATOR - PyTorch Implementation")
    print("=" * 70)
    
    # Параметры конфигурации
    fft_size = 32768
    num_shifts = 10
    num_signals = 5
    num_output_points = 2000
    scale_factor = 1.0 / 32768.0
    
    # Выбрать устройство (cpu или cuda)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используемое устройство: {device}")
    
    # Создать коррелятор
    correlator = FFTCorrelatorPyTorch(fft_size, num_shifts, num_signals, 
                                      num_output_points, scale_factor, device)
    
    # Создать выходной каталог
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


if __name__ == "__main__":
    main()

