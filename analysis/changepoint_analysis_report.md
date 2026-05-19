# Meta-Learning Threshold Analysis Report
Generated: 2025-10-27 12:17:15

## Analysis Window: 45,000 - 70,000 Training Steps

## Changepoint Detection Results

### Explained Variance Analysis
- **Minimum Variance:** 0.1105
- **At Timestep:** 53,248
- **Status:** CRISIS DETECTED

### Detected Changepoints
**Explained Variance:**
  - Page-Hinkley: [np.int64(51200), np.int64(53248), np.int64(57344), np.int64(65536), np.int64(67584), np.int64(69632)]
  - Bayesian: [np.int64(59392)]

**Total PPO Loss:**
  - Page-Hinkley: [np.int64(51200), np.int64(53248), np.int64(57344), np.int64(63488), np.int64(67584)]

**Entropy Bonus:**
  - Page-Hinkley: [np.int64(51200), np.int64(53248), np.int64(59392), np.int64(65536)]

**Policy Loss:**
  - Page-Hinkley: [np.int64(65536)]
  - Bayesian: [np.int64(55296), np.int64(57344)]

**Value Loss:**
  - Bayesian: [np.int64(55296), np.int64(57344)]

**Clip Fraction:**
  - Page-Hinkley: [np.int64(51200), np.int64(59392), np.int64(65536)]

### Consensus Analysis
- **Most Likely Changepoint Window:** 51,000-52,000
- **Supporting Detections:** 4
- **Confidence:** High

## Meta-Learning Interpretation

Based on the changepoint analysis:

1. **Crisis Phase:** Identified by minimum explained variance
2. **Breakthrough Phase:** Following variance recovery
3. **Consolidation Phase:** Stable metrics post-breakthrough

The 55,000-step threshold appears to be supported by:
- Multiple changepoint detection algorithms
- Consistent patterns across different metrics
- Statistical significance in variance changes
