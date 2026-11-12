"""Print comparison table for Run 1 vs Run 2."""
run1 = [2.1647, 1.8866, 1.4004, 1.2994, 1.0263, 0.9621, 0.9201, 0.9569, 0.7265, 1.0834,
        0.7452, 1.1010, 0.9506, 1.1288, 1.3152, 0.9807, 0.8843, 0.9481, 1.0585, 0.9663]
run2 = [1.0885, 0.8275, 1.0942, 0.8003, 0.8540, 0.8229, 0.7438, 0.9463, 0.8255, 1.0712,
        0.7783, 0.8531, 1.1621, 0.9107, 0.9488, 1.2558, 1.0817, 1.1423, 0.6947, 0.7626]
times = [74.3, 40.2, 72.1, 60.4, 65.6, 55.2, 36.2, 43.6, 52.2, 84.3,
         72.8, 36.4, 71.8, 44.1, 75.8, 75.9, 58.4, 54.3, 37.0, 50.0]

print("{:>6} | {:>10} | {:>10} | {:>10} | {:>10}".format(
    "Step", "Run1 Loss", "Run2 Loss", "Delta", "Run2 Time"))
print("-" * 62)
for i in range(20):
    r1, r2 = run1[i], run2[i]
    delta = r2 - r1
    sign = "UP" if delta > 0 else "DOWN"
    print("{:>6} | {:>10.4f} | {:>10.4f} | {:>+9.4f} {} | {:>9.1f}s".format(
        i + 1, r1, r2, delta, sign, times[i]))

avg_r1 = sum(run1) / 20
avg_r2 = sum(run2) / 20
best_r1 = min(run1)
best_r2 = min(run2)
print("-" * 62)
print("{:>6} | {:>10.4f} | {:>10.4f} | {:>+9.4f} |".format(
    "AVG", avg_r1, avg_r2, avg_r2 - avg_r1))
print("{:>6} | {:>10.4f} | {:>10.4f} | {:>+9.4f} |".format(
    "BEST", best_r1, best_r2, best_r2 - best_r1))
print("\nRun 2 avg step time: {:.1f}s  |  Total: {:.0f}s".format(
    sum(times) / len(times), sum(times)))

improved = sum(1 for r1, r2 in zip(run1, run2) if r2 < r1)
print("\nRun 2 improved in {}/20 steps vs Run 1".format(improved))
print("Run 1 loss range: {:.4f} - {:.4f}".format(min(run1), max(run1)))
print("Run 2 loss range: {:.4f} - {:.4f}".format(min(run2), max(run2)))