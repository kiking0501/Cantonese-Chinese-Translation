import os
import configuration as config

# Output Settings
BLEU_OUTPUT_FILE = "as-it-is_baseline.BLEU"

# Test Settings
TEST_INPUT_CHAR = "valid.stdch.sent.tok.char"
TEST_GT_CHAR = "valid.canto.sent.tok.char"

if __name__ == '__main__':
    baseline_dir = os.path.join(config.data_dir, "baselines")
    if not os.path.exists(baseline_dir):
        os.mkdir(baseline_dir)

    testInput = os.path.join(config.data_dir, TEST_INPUT_CHAR)
    testGT = os.path.join(config.data_dir, TEST_GT_CHAR)
    asItIs = testInput

    print("perl multi-bleu.perl -lc " + testGT + " < " + asItIs)
    BLEUOutput = os.popen("perl multi-bleu.perl -lc " + testGT + " < " + asItIs).read()
    with open(os.path.join(baseline_dir, BLEU_OUTPUT_FILE), "w") as f:
        print(BLEUOutput)
        f.write(BLEUOutput)
