# check_normalization.py
from combined_preprocessing import check_unk_rate

# Check train split
print("\nChecking TRAIN split...")
train_stats = check_unk_rate(split='train')

# Check test split
print("\nChecking TEST split...")
test_stats = check_unk_rate(split='test')

# Print summary
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Train <unk> rate: {train_stats['unk_rate']:.2f}%")
print(f"Test <unk> rate: {test_stats['unk_rate']:.2f}%")

if train_stats['unk_rate'] < 1.0 and test_stats['unk_rate'] < 1.0:
    print("\n✅ Normalization is working well!")
else:
    print("\n⚠️ High <unk> rate detected. Review samples and add synonyms.")