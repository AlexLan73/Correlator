# 📌 CORRELATOR OPTIMIZATION - QUICK REMINDER

> **For next chat session:** Just open this file to remember what we were doing!

---

## 🎯 WHAT WE DISCOVERED (2025-12-25)

**Hardware Test Results:**
```
RTX 3060 Capabilities:
✅ OpenCL 3.0 supported
❌ Command Buffers NOT available (NVIDIA limitation)
⚠️  SVM coarse-grain available
```

**Performance Bottleneck Found:**
```
Current timing: ~95 ms per cycle
Problem: clWaitForEvents() blocks CPU 9-24 ms per cycle! 💥

Solution: Remove clWaitForEvents() → +16% faster (80 ms)
Better: Add SVM → +30% faster (65 ms) on RTX 3060
```

---

## 🚀 OPTIMIZATION PLAN (2 PHASES)

### ⭐ PHASE 1 (15 MINUTES) - DO THIS FIRST!

**Task:** Remove `clWaitForEvents()` from GPU pipeline

**Where to change:**
- `src/fft_handler.cpp` - line 940 and similar locations
- Search for all `clWaitForEvents(1, &event_`

**What to do:**
```cpp
// BEFORE:
clEnqueueWriteBuffer(..., &event_upload);
clWaitForEvents(1, &event_upload);  // ❌ DELETE THIS LINE

// AFTER:
clEnqueueWriteBuffer(..., nullptr);  // ✅ CHANGE &event TO nullptr

// At END of executeFullPipeline():
clFinish(ctx_.queue);  // ✅ ADD THIS - single wait at end
```

**Expected Result:** 10-20% speedup (16% predicted)

**Implementation Checklist:**
- [ ] Find all clWaitForEvents() calls
- [ ] Replace with nullptr argument
- [ ] Add clFinish() at pipeline end
- [ ] Test compilation
- [ ] Run benchmark
- [ ] Compare timing: was ~95ms, should be ~80ms now

---

### ⭐⭐ PHASE 2 (2-3 HOURS) - ONLY ON RTX 3060

**Task:** Use SVM (Shared Virtual Memory) for input buffers

**Why:** GPU reads from host memory without PCIe transfer overhead

**Where to change:**
- `src/fft_handler.cpp` - buffer allocation and WriteBuffer calls
- `include/correlator/OpenCLFFTBackend.hpp` - member variables

**What to do:**

1. **Change allocation:**
```cpp
// BEFORE:
ctx_.input_data = clCreateBuffer(ctx_.context, CL_MEM_READ_WRITE, size, nullptr, &err);

// AFTER:
int32_t* ctx_.input_data_svm = (int32_t*)clSVMAlloc(
    ctx_.context,
    CL_MEM_READ_WRITE | CL_MEM_SVM_COARSE_GRAIN_BUFFER,
    size, 0);
```

2. **Change data transfer:**
```cpp
// BEFORE:
clEnqueueWriteBuffer(ctx_.queue, ctx_.input_data, CL_FALSE, 0, size, host_input, ...);

// AFTER:
memcpy(ctx_.input_data_svm, host_input, size);

// Add explicit sync for coarse-grain:
clEnqueueSVMMap(ctx_.queue, CL_FALSE, CL_MAP_WRITE, ctx_.input_data_svm, size, 0, nullptr, nullptr);
clEnqueueSVMUnmap(ctx_.queue, ctx_.input_data_svm, 0, nullptr, nullptr);
```

3. **FFT uses SVM pointer directly**

**Note:** RTX 2080 Ti doesn't support SVM - add fallback to regular buffers

**Expected Result:** Additional 5-15% speedup (total 20-35% from baseline)

**Implementation Checklist:**
- [ ] Create SVM allocation wrapper
- [ ] Replace input_data buffer allocation
- [ ] Replace WriteBuffer with memcpy
- [ ] Add SVM map/unmap for sync
- [ ] Add RTX 2080 Ti fallback (regular buffers)
- [ ] Test on RTX 3060 (Phase 1 only on RTX 2080 Ti)
- [ ] Compare timing: should be ~65ms now

---

## 📊 EXPECTED TIMELINE

```
Before any optimization:    95 ms per cycle
After PHASE 1:              80 ms per cycle  (+16% speedup)
After PHASE 2 (3060 only):  65 ms per cycle  (+30% total speedup)
```

---

## 🔧 KEY FILES TO MODIFY

```
correlator/
├── src/
│   └── fft_handler.cpp          ← MAIN CHANGES HERE
│       Line ~914: clEnqueueWriteBuffer() + clWaitForEvents()
│       Line ~940: Remove clWaitForEvents()
│       Line ~960: Add clFinish() at end
│
├── include/
│   ├── correlator/
│   │   ├── CorrelationPipeline.hpp   ← Add final clFinish()
│   │   └── OpenCLFFTBackend.hpp      ← SVM pointers (Phase 2)
```

---

## 💡 WHY THIS WORKS

### Current Problem:
```
CPU: clEnqueueWriteBuffer() → clWaitForEvents() BLOCKS!
     (GPU is processing, CPU waits 3-8 ms)
     clEnqueueTransform() 
     clWaitForEvents() BLOCKS!
     (GPU is processing, CPU waits 3-8 ms)
     clEnqueueReadBuffer()
     clWaitForEvents() BLOCKS!
     (GPU is processing, CPU waits 3-8 ms)

Total CPU blocking: 9-24 ms 💥
```

### Solution - Phase 1:
```
CPU: clEnqueueWriteBuffer() (1-2 µs, no wait!)
     clEnqueueTransform() (no wait!)
     clEnqueueReadBuffer() (no wait!)
     clFinish() (wait only once at end)

GPU sees full command stream at once → better optimization!
Total CPU blocking: ~0 ms ✅
Gain: 9-24 ms = 16% speedup
```

### Solution - Phase 2 (RTX 3060):
```
Replace PCIe WriteBuffer with SVM memcpy:
- memcpy faster than GPU-aware transfer
- GPU prefetches via hardware cache
- No explicit synchronization overhead

Additional gain: 5-15%
```

---

## ⚠️ IMPORTANT NOTES

1. **Phase 1 works on BOTH cards** (RTX 2080 Ti + RTX 3060)
2. **Phase 2 only for RTX 3060** (RTX 2080 Ti has no SVM)
3. **Test correctness first:** Peak values must match baseline
4. **Measure both cards:** Ensure no regression on RTX 2080 Ti
5. **Use nvidia-smi:** Monitor GPU during optimization

---

## 🎯 SUCCESS CRITERIA

✅ **Phase 1 Done When:**
- All clWaitForEvents() removed between operations
- Single clFinish() at pipeline end
- Code compiles without errors
- Timing: ~80 ms (was 95 ms) ✓
- Peak values identical to baseline ✓

✅ **Phase 2 Done When (RTX 3060):**
- SVM buffers allocated and working
- memcpy replaces WriteBuffer
- SVM map/unmap synchronization correct
- Timing: ~65 ms (was 95 ms) ✓
- Fallback to regular buffers on RTX 2080 Ti ✓

---

## 📋 REFERENCE DOCUMENTS

Created in this session:

1. **OPTIMIZATION_ROADMAP.md** - Full detailed plan
2. **rtx3060_analysis.md** - Hardware analysis & numbers
3. **COMMAND_BUFFERS_STRATEGY.md** - Why CB not available
4. **SVM_GPU_QUEUE_LATENCY.md** - How SVM reduces latency
5. **opencl_capability_checker.cpp** - Hardware detection tool

---

## 🚀 QUICK START FOR NEXT SESSION

1. Open **OPTIMIZATION_ROADMAP.md** for full context
2. Read this file for quick reminder
3. Check Phase 1 checklist
4. Implement changes in fft_handler.cpp
5. Measure timing improvement
6. Decide on Phase 2

---

## 📞 NEXT STEPS

When continuing in new chat:

1. **Say:** "Continue with correlator optimization from 2025-12-25"
2. **Reference:** This file + OPTIMIZATION_ROADMAP.md
3. **Confirm:** Hardware (RTX 3060 + RTX 2080 Ti)
4. **Start:** Phase 1 implementation (15 minutes)

---

**Remember:** 
- 🎯 Goal: 30% speedup (16% Phase 1 + 15% Phase 2)
- ⏱️ Time: ~3 hours total work
- 🔧 Effort: LOW to MEDIUM complexity
- 📈 Impact: HIGH (every corellation 30ms faster!)

**Let's ship it! 🚀**

---

**Created:** 2025-12-25 10:30 AM MSK  
**Status:** Ready to implement  
**Next Action:** Open new chat → Reference this file
