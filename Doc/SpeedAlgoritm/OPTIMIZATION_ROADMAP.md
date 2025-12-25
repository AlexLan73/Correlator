# 🚀 CORRELATOR OPTIMIZATION ROADMAP

## 📋 PROJECT SUMMARY

**Project:** Real-time GPU FFT Correlator with Signal Processing  
**Owner:** Alex (Alexan73)  
**Hardware:** RTX 2080 Ti + RTX 3060  
**Current Status:** 3-step pipeline working, ready for optimization  
**Last Updated:** 2025-12-25

---

## 🎯 OPTIMIZATION GOALS

| Goal | Current | Target | Gain |
|------|---------|--------|------|
| **STEP 1** | ~20 ms | ~15 ms | 25% |
| **STEP 2** | ~30 ms | ~25 ms | 17% |
| **STEP 3** | ~45 ms | ~40 ms | 11% |
| **TOTAL** | ~95 ms | ~80 ms | **16%** 🚀 |
| **With SVM** | ~95 ms | ~65 ms | **30%** 🚀🚀 |

---

## 🔧 OPTIMIZATION STRATEGIES

### Strategy 1: Asynchronous Enqueue (PRIORITY 1 ⭐⭐⭐)

**Hardware Support:** ✅ RTX 2080 Ti, ✅ RTX 3060  
**Complexity:** MINIMAL (15 minutes)  
**Expected Gain:** 10-20% overall speedup  
**Risk Level:** LOW (proven approach)  

#### What to do:
1. Remove all `clWaitForEvents()` between STEPS
2. Add single `clFinish()` at the end of pipeline
3. Let GPU scheduler batch commands

#### Key Changes:
```cpp
// BEFORE:
clEnqueueWriteBuffer(..., &event);
clWaitForEvents(1, &event);      // ❌ REMOVE THIS!
clfftEnqueueTransform(...);

// AFTER:
clEnqueueWriteBuffer(..., nullptr);  // ✅ nullptr instead of &event
clfftEnqueueTransform(...);          // ✅ Immediate enqueue
// ... more operations ...
clFinish(queue);                      // ✅ Single wait at end
```

#### Files to Modify:
- `src/fft_handler.cpp` - remove clWaitForEvents() calls
- `include/correlator/CorrelationPipeline.hpp` - add final clFinish()

#### Why it works:
- Current: CPU blocks on EVERY WriteBuffer (9-24 ms total blocking)
- After: GPU queues all commands, executes in batch (0 ms blocking)
- GPU scheduler has visibility into full command stream

---

### Strategy 2: SVM (Shared Virtual Memory) (PRIORITY 2 ⭐⭐)

**Hardware Support:** ✅ RTX 3060 (coarse-grain), ❌ RTX 2080 Ti  
**Complexity:** MEDIUM (2-3 hours)  
**Expected Gain:** Additional 5-15% speedup (total 20-35%)  
**Risk Level:** MEDIUM (SVM can be quirky on NVIDIA)  

#### What to do:
1. Replace host buffer allocation with SVM allocation
2. Replace `clEnqueueWriteBuffer()` with `memcpy()`
3. Add explicit SVM map/unmap for coarse-grain synchronization
4. GPU reads directly from SVM (no explicit transfer)

#### Key Changes:
```cpp
// BEFORE:
cl_mem ctx_.input_data = clCreateBuffer(
    ctx_.context, CL_MEM_READ_WRITE,
    size, nullptr, &err);

clEnqueueWriteBuffer(ctx_.queue, ctx_.input_data, CL_FALSE, 0,
                     size, host_input, 0, nullptr, nullptr);

// AFTER:
int32_t* svm_input = (int32_t*)clSVMAlloc(
    ctx_.context,
    CL_MEM_READ_WRITE | CL_MEM_SVM_COARSE_GRAIN_BUFFER,
    size, 0);

memcpy(svm_input, host_input, size);  // ✅ No GPU transfer overhead

// Explicit flush for coarse-grain:
clEnqueueSVMMap(ctx_.queue, CL_FALSE, CL_MAP_WRITE,
                svm_input, size, 0, nullptr, nullptr);
clEnqueueSVMUnmap(ctx_.queue, svm_input, 0, nullptr, nullptr);

// GPU reads from SVM:
clfftEnqueueTransform(ctx_.queue, ..., svm_input, ...);
```

#### Files to Modify:
- `src/fft_handler.cpp` - SVM allocation and operations
- `include/correlator/OpenCLFFTBackend.hpp` - SVM pointer management

#### Why it works:
- Coarse-grain SVM still supported on RTX 3060
- Eliminates PCIe transfer overhead for input buffers
- GPU can prefetch SVM data during computation

#### Compatibility:
- RTX 3060: WORKS (coarse-grain SVM available)
- RTX 2080 Ti: SKIP (no SVM support, use Strategy 1 only)

---

### Strategy 3: Command Buffers (PRIORITY 3 - NOT APPLICABLE)

**Hardware Support:** ❌ RTX 2080 Ti, ❌ RTX 3060  
**Complexity:** HIGH  
**Expected Gain:** Would be 10-25% (if supported)  
**Risk Level:** HIGH (not supported on NVIDIA)  

**STATUS:** ❌ NOT RECOMMENDED
- OpenCL 3.0 available but `cl_khr_command_buffer` extension missing
- Both cards don't expose command buffer capability
- Use Strategy 1 + Strategy 2 instead

---

## 📊 CURRENT CODE STRUCTURE

### Algorithm Overview:
```
STEP 1: Reference Signal Processing
├─ clEnqueueWriteBuffer(reference)
├─ clEnqueueTransform(FFT forward)        [32 correlators]
└─ Timing: ~20 ms

STEP 2: Input Signals Processing
├─ clEnqueueWriteBuffer(input)            [50 signals]
├─ clEnqueueTransform(FFT forward)
└─ Timing: ~30 ms

STEP 3: Correlation & Results
├─ clEnqueueTransform(complex multiply + IFFT)  [1600 pairs]
├─ clEnqueueReadBuffer(results)           [2000 peaks]
└─ Timing: ~45 ms

TOTAL: ~95 ms per cycle
```

### Key Files:
- `src/fft_handler.cpp` (914 lines) - Core FFT and GPU memory management
- `include/correlator/CorrelationPipeline.hpp` - 3-step pipeline orchestration
- `include/correlator/OpenCLFFTBackend.hpp` - OpenCL backend wrapper

### Current Bottlenecks:
1. **clWaitForEvents() calls** between steps (9-24 ms CPU blocking)
2. **No SVM utilization** (uses regular host-pinned buffers)
3. **Sequential step execution** (GPU scheduler sees one command at a time)

---

## 🎯 IMPLEMENTATION ROADMAP

### Phase 1: Async Enqueue (IMMEDIATE - 1 HOUR)

```
┌─────────────────────────────────────┐
│ Task: Remove clWaitForEvents()      │
├─────────────────────────────────────┤
│ 1. Locate clWaitForEvents() calls   │
│ 2. Remove event waiting between ops │
│ 3. Add clFinish() at pipeline end   │
│ 4. Test for correctness             │
│ 5. Benchmark improvement            │
└─────────────────────────────────────┘
Expected Gain: 10-20%
```

**Checklist:**
- [ ] Found all `clWaitForEvents()` in fft_handler.cpp
- [ ] Replaced `&event_upload` with `nullptr`
- [ ] Replaced `&event_callback` with `nullptr`
- [ ] Replaced `&event_download` with `nullptr`
- [ ] Added `clFinish(ctx_.queue)` before final results return
- [ ] Compiled without errors
- [ ] Tested with same parameters (2^19, 32 shifts, 50 signals, 5 beams)
- [ ] Measured timing improvement
- [ ] Committed to git

---

### Phase 2: SVM Integration (NEXT - 2-3 HOURS)

**Only for RTX 3060!**

```
┌─────────────────────────────────────┐
│ Task: Integrate Coarse-Grain SVM    │
├─────────────────────────────────────┤
│ 1. Create SVM allocation wrapper    │
│ 2. Replace input buffer with SVM    │
│ 3. Replace WriteBuffer with memcpy  │
│ 4. Add SVM map/unmap for sync       │
│ 5. Test on both cards (3060 only)   │
│ 6. Benchmark additional gain        │
└─────────────────────────────────────┘
Expected Additional Gain: 5-15%
Total: 20-35% overall improvement
```

**Checklist:**
- [ ] Created `SVMBufferWrapper` class
- [ ] Allocated input_data as SVM (coarse-grain)
- [ ] Replaced clEnqueueWriteBuffer with memcpy
- [ ] Added clEnqueueSVMMap/Unmap for flush
- [ ] Added fallback for RTX 2080 Ti (regular buffers)
- [ ] Compiled for both cards
- [ ] Tested on RTX 3060
- [ ] Measured additional improvement
- [ ] Committed to git

---

## 🔍 MONITORING & VALIDATION

### Performance Metrics to Track:

```cpp
// Per pipeline execution:
double timing_step1_upload;      // clEnqueueWriteBuffer overhead
double timing_step1_fft;         // FFT compute
double timing_step2_upload;      // clEnqueueWriteBuffer overhead
double timing_step2_fft;         // FFT compute
double timing_step3_multiply;    // Complex multiply + IFFT
double timing_step3_download;    // clEnqueueReadBuffer overhead
double timing_gpu_wait;          // clFinish or clWaitForEvents total
double timing_total;             // End-to-end

// Expected improvements:
// Before: timing_gpu_wait = 9-24 ms
// After Phase 1: timing_gpu_wait ≈ 0 ms
// After Phase 2: timing_step2_upload, timing_step3_download reduced
```

### Validation Checklist:

```cpp
// Correctness validation:
- Peak correlation values match pre-optimization
- Number of peaks matches expected (2000 per beam)
- Beam indices are identical to baseline
- No GPU errors (CL_SUCCESS on all operations)
- No data races or synchronization issues

// Performance validation:
- Overall speedup matches predicted (10-20% Phase 1, +5-15% Phase 2)
- No slowdown on RTX 2080 Ti (Phase 1 only)
- Memory usage unchanged
- Power consumption acceptable
```

---

## 📈 EXPECTED RESULTS

### Baseline (Current):
- Total time: ~95 ms per cycle
- CPU blocking: 9-24 ms on GPU operations
- PCIe transfers: ~105 MB (input signals)

### After Phase 1 (Async Enqueue):
- Total time: ~80 ms per cycle  ✅ **16% faster**
- CPU blocking: ~0 ms (GPU executes queued commands)
- PCIe transfers: unchanged
- Implementation: 15 minutes

### After Phase 2 (SVM on RTX 3060):
- Total time: ~65 ms per cycle  ✅ **30% faster total**
- CPU blocking: 0 ms
- PCIe transfers: reduced (SVM avoids explicit transfer)
- Implementation: additional 2-3 hours (RTX 3060 only)

### Cumulative Benefit:
```
Baseline:              95 ms
Phase 1:               80 ms  (-15 ms, -16%)
Phase 2:               65 ms  (-30 ms, -30% from baseline)

Sustained improvement per beam: 30+ ms saved
Annual throughput increase: ~400 more correlations per second
```

---

## 🛠️ REFERENCE DOCUMENTATION

### Hardware Capabilities:
- **RTX 3060:** `opencl_capability_checker` output (2025-12-25)
  - OpenCL 3.0 ✅
  - SVM Coarse-Grain ✅
  - Command Buffers ❌
  - Strategy: Phase 1 + Phase 2

- **RTX 2080 Ti:** Expected OpenCL 1.2
  - OpenCL 3.0 ❌
  - SVM ❌ (probably)
  - Command Buffers ❌
  - Strategy: Phase 1 only

### Key OpenCL Concepts:
- `clWaitForEvents()` - CPU blocking until GPU completes event
- `clFinish()` - CPU blocking until entire queue completes
- `clSVMAlloc()` - Shared Virtual Memory allocation
- Coarse-grain SVM - Requires explicit map/unmap synchronization
- Command Buffers - Pre-recorded command graphs (not available)

### Related Files:
- `COMMAND_BUFFERS_STRATEGY.md` - Analysis (not applicable)
- `SVM_GPU_QUEUE_LATENCY.md` - SVM explanation
- `opencl_capability_checker.cpp` - Hardware capability detection
- `fft_handler.cpp` - Main implementation (914 lines)

---

## 📝 NOTES & OBSERVATIONS

### Why This Optimization Works:

1. **Async Enqueue eliminates CPU blocking:**
   - Current: CPU waits 3-8 ms per WriteBuffer for GPU to accept command
   - With async: CPU adds command (1-2 µs) and continues
   - GPU batch-executes: can optimize order, reduce context switches

2. **SVM reduces memory transfer overhead:**
   - Current: CPU → PCIe → GPU (bandwidth-limited)
   - With SVM: CPU writes to host buffer, GPU prefetches via cache
   - Coarse-grain sufficient for batched operations

3. **GPU scheduler benefits from visibility:**
   - Current: Scheduler sees 1 command at a time, sub-optimal ordering
   - After async: Scheduler sees queue of commands, can parallelize
   - DMA can overlap with compute within single batch

### Potential Issues & Mitigations:

| Issue | Mitigation |
|-------|-----------|
| **Async commands may complete out-of-order** | Use correct dependency chains with clEnqueueBarrier if needed |
| **SVM coarse-grain adds map/unmap overhead** | Minimal compared to WriteBuffer savings |
| **GPU memory pressure if commands queue too much** | clFinish() provides safety valve |
| **Debugging becomes harder with async** | Add detailed timing instrumentation in each STEP |

---

## 🎯 SUCCESS CRITERIA

✅ **Phase 1 Complete When:**
- All clWaitForEvents() removed from pipeline
- Final clFinish() added
- Code compiles without warnings
- Functionality identical to baseline
- Measured speedup: 10-20% (target: 16%)

✅ **Phase 2 Complete When (RTX 3060 only):**
- SVM allocation working for input buffers
- memcpy replaces WriteBuffer for input
- SVM map/unmap synchronization correct
- Fallback to regular buffers on RTX 2080 Ti
- Additional measured speedup: 5-15%
- Total speedup: 20-35%

✅ **Project Success Criteria:**
- Overall speedup: minimum 15%, target 30%
- No functionality loss (peak values unchanged)
- Both hardware platforms supported
- Code remains maintainable and documented
- Performance gain stable across multiple runs

---

## 📞 FOLLOW-UP ITEMS

After next chat session:

1. **Check GPU Load:** Use `nvidia-smi` to verify GPU isn't bottlenecking elsewhere
2. **Profile Individual Steps:** Add more granular timing for each STEP
3. **Test Edge Cases:** 
   - Minimum signal length (2^10)
   - Maximum correlators (64)
   - Different beam counts (1-10)
4. **Compare with CUDA:** If curious, benchmark equivalent CUDA code
5. **Document Lessons:** Write blog post about GPU optimization pipeline

---

## 🚀 NEXT SESSION AGENDA

When opening a new chat, reference this file and:

1. Confirm Phase 1 implementation status
2. Review measured performance improvements
3. Decide on Phase 2 (SVM) based on time availability
4. Plan testing on both RTX 2080 Ti and RTX 3060
5. Discuss any issues encountered

---

**Created:** 2025-12-25 10:30 AM MSK  
**Project Status:** Ready for Phase 1 Implementation  
**Estimated Time to 30% Speedup:** 3-4 hours total work  
**Risk Level:** LOW (well-understood optimization techniques)

**Let's make the Correlator fly! 🚀✨**
