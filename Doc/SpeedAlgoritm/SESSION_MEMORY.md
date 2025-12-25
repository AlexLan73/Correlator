# 🎯 CORRELATOR PROJECT - SESSION MEMORY FILE

> **USE THIS FILE:** When starting new chat, attach this file and say:  
> "I'm continuing GPU correlator optimization. Here's what we discovered."

---

## 📋 PROJECT INFO

| Параметр | Значение |
|----------|----------|
| **Project** | Real-time GPU FFT Correlator |
| **Owner** | Alex (Alexan73) |
| **Hardware** | RTX 2080 Ti + RTX 3060 |
| **Framework** | OpenCL + CUDA |
| **Session Date** | 2025-12-25 |
| **Status** | Algorithm working, optimizing |

---

## 🔍 KEY DISCOVERIES

### Hardware Capabilities Test Results

**RTX 3060** (2025-12-25):
```
✅ OpenCL 3.0 (modern!)
❌ Command Buffers (cl_khr_command_buffer) - NOT AVAILABLE
⚠️  SVM Coarse-Grain - AVAILABLE
❌ Unified Memory - NOT AVAILABLE
```

**RTX 2080 Ti** (expected):
```
OpenCL 1.2 (older)
No SVM, no Command Buffers
Async approach only
```

### Performance Bottleneck Analysis

```
Current bottleneck: clWaitForEvents() BLOCKS CPU!

Timeline:
Step 1: clWaitForEvents() ............. 3-8 ms BLOCKING
Step 2: clWaitForEvents() ............. 3-8 ms BLOCKING
Step 3: clWaitForEvents() ............. 3-8 ms BLOCKING
                                        ────────────
                        Total blocking: 9-24 ms per cycle! 💥

This is 10-25% of total ~95ms cycle time!
```

### Solution Strategies

| Strategy | RTX 2080 Ti | RTX 3060 | Speedup | Effort |
|----------|---------|---------|---------|--------|
| **Remove clWaitForEvents()** | ✅ | ✅ | +16% | 15 min |
| **Add SVM** | ❌ | ✅ | +15% more | 2-3 hrs |
| **Command Buffers** | ❌ | ❌ | N/A | Not available |

---

## 📊 ALGORITHM STRUCTURE

### Current 3-Step Pipeline

```
┌─────────────────────────────────────────────────────┐
│                 STEP 1                              │
│  Upload reference signal (1×N)                      │
│  FFT forward (32 correlators)                       │
│  clWaitForEvents() here ← BLOCKING!                 │
│  Timing: ~20 ms (3-8 ms overhead)                   │
└─────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│                 STEP 2                              │
│  Upload input signals (50×N)                        │
│  FFT forward                                        │
│  clWaitForEvents() here ← BLOCKING!                 │
│  Timing: ~30 ms (3-8 ms overhead)                   │
└─────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│                 STEP 3                              │
│  Complex multiply + IFFT (1600 pairs)               │
│  Download results (2000 peaks per beam)             │
│  clWaitForEvents() here ← BLOCKING!                 │
│  Timing: ~45 ms (3-8 ms overhead)                   │
└─────────────────────────────────────────────────────┘

TOTAL: 95 ms (9-24 ms wasted on CPU blocking!)
```

### Parameters

```
N = 2^19 = 524,288 samples (signal length)
num_signals = 50 (input signals)
num_shifts = 32 (correlators per beam)
n_kg = 5 (beams)
Total correlations: 50 × 32 = 1600
```

### Key Files

```
fft_handler.cpp
  - Line 914-960: STEP 2 (input upload + FFT)
  - Contains clEnqueueWriteBuffer() + clWaitForEvents()
  - Also in STEP 1 and STEP 3

CorrelationPipeline.hpp
  - Lines 213-220: executeFullPipeline()
  - Orchestrates 3 steps sequentially
```

---

## 🚀 OPTIMIZATION ROADMAP

### PHASE 1: Remove clWaitForEvents() (15 minutes) ⭐⭐⭐

**Current Code Pattern:**
```cpp
err = clEnqueueWriteBuffer(ctx_.queue, ctx_.input_data, CL_FALSE, 0,
                          N * num_signals * sizeof(int32_t),
                          host_input, 0, nullptr, &event_upload);
if (err != CL_SUCCESS) throw std::runtime_error("...");

clWaitForEvents(1, &event_upload);  // ❌ BLOCKS HERE!

err = clfftEnqueueTransform(...);
if (err != CL_SUCCESS) throw std::runtime_error("...");
```

**Optimized Pattern:**
```cpp
err = clEnqueueWriteBuffer(ctx_.queue, ctx_.input_data, CL_FALSE, 0,
                          N * num_signals * sizeof(int32_t),
                          host_input, 0, nullptr, nullptr);  // ✅ NO WAIT!
if (err != CL_SUCCESS) throw std::runtime_error("...");

// Remove clWaitForEvents() call entirely!

err = clfftEnqueueTransform(...);  // ✅ Immediate enqueue
if (err != CL_SUCCESS) throw std::runtime_error("...");

// Later, at END of executeFullPipeline():
clFinish(ctx_.queue);  // ✅ Single wait at end
```

**Expected Results:**
- Remove 9-24 ms CPU blocking per cycle
- Speedup: 16% (95 ms → 80 ms)
- GPU scheduler batches commands better
- No functional change, same results

**Checklist:**
- [ ] Search fft_handler.cpp for "clWaitForEvents"
- [ ] For each occurrence between operations: remove it
- [ ] Change `&event_upload` → `nullptr` in clEnqueueWriteBuffer
- [ ] Add `clFinish(ctx_.queue)` before returning from pipeline
- [ ] Compile and test
- [ ] Benchmark: measure timing improvement
- [ ] Verify: peak values unchanged

---

### PHASE 2: Add SVM (2-3 hours) ⭐⭐ (RTX 3060 only)

**Buffer Allocation Change:**
```cpp
// BEFORE:
ctx_.input_data = clCreateBuffer(
    ctx_.context, CL_MEM_READ_WRITE,
    num_signals * N * sizeof(int32_t), nullptr, &err);

// AFTER:
int32_t* ctx_.input_data_svm = (int32_t*)clSVMAlloc(
    ctx_.context,
    CL_MEM_READ_WRITE | CL_MEM_SVM_COARSE_GRAIN_BUFFER,
    num_signals * N * sizeof(int32_t), 0);
```

**Memory Transfer Change:**
```cpp
// BEFORE:
clEnqueueWriteBuffer(ctx_.queue, ctx_.input_data, CL_FALSE, 0,
                    size, host_input, 0, nullptr, nullptr);

// AFTER:
memcpy(ctx_.input_data_svm, host_input, size);  // ✅ Host-side copy

// Explicit sync for coarse-grain SVM:
clEnqueueSVMMap(ctx_.queue, CL_FALSE, CL_MAP_WRITE,
               ctx_.input_data_svm, size, 0, nullptr, nullptr);
clEnqueueSVMUnmap(ctx_.queue, ctx_.input_data_svm, 0, nullptr, nullptr);

// GPU reads from SVM automatically
clfftEnqueueTransform(ctx_.queue, ..., ctx_.input_data_svm, ...);
```

**RTX 2080 Ti Fallback:**
```cpp
// Detect SVM support:
cl_device_svm_capabilities svm_caps = 0;
clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES, sizeof(svm_caps), &svm_caps, nullptr);

if (svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) {
    // Use SVM path (RTX 3060)
} else {
    // Use regular buffer path (RTX 2080 Ti)
}
```

**Expected Results:**
- Additional 5-15% speedup (total 20-35% from baseline)
- Timing: 95 ms → 65 ms
- Only works on RTX 3060
- RTX 2080 Ti continues with PHASE 1 only

**Checklist:**
- [ ] Create SVM buffer wrapper class
- [ ] Allocate input_data as SVM (coarse-grain)
- [ ] Replace WriteBuffer with memcpy
- [ ] Add SVM map/unmap for synchronization
- [ ] Add device capability detection for fallback
- [ ] Compile for both RTX 2080 Ti and RTX 3060
- [ ] Test on RTX 3060 for new path
- [ ] Test on RTX 2080 Ti for fallback
- [ ] Benchmark both: measure additional gain
- [ ] Verify: peak values unchanged on both

---

## 📈 EXPECTED PERFORMANCE GAINS

```
Baseline (current):             95 ms per cycle
After PHASE 1 (both cards):     80 ms per cycle  (+16%)
After PHASE 2 (RTX 3060 only):  65 ms per cycle  (+30% total)

Per-beam improvement:
  - STEP 1: 20 ms → 15 ms
  - STEP 2: 30 ms → 25 ms (PHASE 1) → 20 ms (PHASE 2)
  - STEP 3: 45 ms → 40 ms (PHASE 1) → 35 ms (PHASE 2)

Over 5 beams × multiple correlations:
  - Baseline: 475 ms per full correlation
  - After all optimizations: 325 ms = 32% faster!
```

---

## 🔧 FILES TO MODIFY

### PHASE 1 Changes:

**File: src/fft_handler.cpp**
- Search for all `clWaitForEvents` occurrences
- Around line 940: clEnqueueWriteBuffer + clWaitForEvents for input
- Similar patterns in STEP 1 and STEP 3
- Remove blocking wait calls
- Add clFinish at pipeline end

**File: include/correlator/CorrelationPipeline.hpp**
- Line 213-220: executeFullPipeline()
- Add clFinish(ctx_.queue) at end before return

### PHASE 2 Changes (RTX 3060 only):

**File: src/fft_handler.cpp**
- Buffer allocation (change clCreateBuffer to clSVMAlloc)
- Memory transfer (change clEnqueueWriteBuffer to memcpy)
- Add SVM map/unmap synchronization

**File: include/correlator/OpenCLFFTBackend.hpp**
- struct OpenCLContext members
- Add SVM pointer alongside or instead of regular mem

---

## 💡 TECHNICAL DETAILS

### Why clWaitForEvents() Blocks

```
1. CPU calls clEnqueueWriteBuffer() - adds command to GPU queue
2. CPU calls clWaitForEvents() - blocks until GPU executes command
3. GPU busy with other work, WriteBuffer waits in queue
4. When GPU finally processes it: DMA takes 3-8 ms
5. GPU signals completion
6. CPU wakes up from clWaitForEvents()

Problem: CPU idle during steps 3-5 (3-8 ms lost!)
```

### Why Async Helps

```
1. CPU calls clEnqueueWriteBuffer() - adds command to GPU queue
2. CPU immediately calls clfftEnqueueTransform() - adds more commands
3. CPU can continue (doesn't wait)
4. GPU scheduler sees multiple commands and optimizes execution
5. GPU executes efficiently: maybe DMA overlaps with other work
6. CPU calls clFinish() only when actually needs results

Benefit: GPU sees full pipeline, better scheduler decisions
```

### Why SVM Helps (on RTX 3060)

```
1. CPU writes to SVM buffer (host memory) - very fast
2. GPU prefetches from SVM (hardware cache helps)
3. No explicit PCIe DMA transfer overhead
4. Coarse-grain SVM: map/unmap still needed for safety
5. But cheaper than full WriteBuffer

Benefit: Host-GPU communication faster, less PCIe congestion
```

---

## 📚 REFERENCE DOCUMENTS

All created in this session:

1. **OPTIMIZATION_ROADMAP.md** ← MAIN REFERENCE
   - Full detailed plan with code examples
   - Success criteria and validation
   - Risk mitigation strategies

2. **OPTIMIZATION_QUICK_REMINDER.md**
   - Quick reference for next session
   - Checklists and key points
   - File locations

3. **rtx3060_analysis.md**
   - Hardware analysis
   - Detailed performance calculations
   - Why each strategy works

4. **SVM_GPU_QUEUE_LATENCY.md**
   - Deep dive into SVM mechanics
   - How SVM reduces latency
   - When to use what approach

5. **COMMAND_BUFFERS_STRATEGY.md**
   - Why Command Buffers would help (if available)
   - Explanation of command graphs
   - Why RTX 3060 doesn't support it

6. **opencl_capability_checker.cpp**
   - Code to detect hardware capabilities
   - Use to confirm on any GPU

---

## 🎯 NEXT SESSION AGENDA

When opening new chat:

1. **Attach this file** as reference
2. **Say:** "Continuing GPU correlator optimization from 2025-12-25"
3. **Review:** What we discovered (hardware capabilities, bottleneck)
4. **Start:** PHASE 1 implementation (15 minutes)
5. **Test:** Verify 16% speedup achieved
6. **Decide:** Proceed to PHASE 2 or stop at PHASE 1

---

## 🚀 SUCCESS TIMELINE

```
Immediate (15 min):  PHASE 1 done → 95 ms → 80 ms
Today (2-3 hrs more): PHASE 2 on RTX 3060 → 65 ms
Ongoing: Test both cards, verify results

Total improvement: 30% speedup for ~4 hours work
```

---

## ⚠️ CRITICAL REMINDERS

- ✅ PHASE 1 works on BOTH cards (RTX 2080 Ti + RTX 3060)
- ✅ PHASE 2 only for RTX 3060 (RTX 2080 Ti has no SVM)
- ✅ Test correctness: peak values must be IDENTICAL to baseline
- ✅ Measure both cards: ensure no regression
- ✅ Use `nvidia-smi` to monitor GPU during testing

---

**This file is your memory bridge between chat sessions!**

Attachment checklist for next session:
- [ ] This file (SESSION_MEMORY.md)
- [ ] OPTIMIZATION_ROADMAP.md (main plan)
- [ ] RTX3060 analysis results (hardware test output)
- [ ] Current fft_handler.cpp (to show where changes are)

---

**Last Updated:** 2025-12-25 10:30 AM MSK  
**Next Action:** Reference this file in new chat session  
**Status:** Ready to optimize! 🚀
