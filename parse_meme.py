import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import logomaker

# Compute Shannon information content for a PWM column
def column_information(col):
    eps = 1e-9  # to avoid log(0)
    background = 0.25
    return np.sum([p * np.log2((p + eps) / background) for p in col if p > 0])

# Trim low-information flanks
def trim_pwm(pwm, threshold=0.2):
    infos = [column_information(col) for col in pwm]
    
    left = 0
    while left < len(pwm) and infos[left] < threshold:
        left += 1
    
    right = len(pwm) - 1
    while right >= 0 and infos[right] < threshold:
        right -= 1
    
    if left > right:
        return []
    return pwm[left:right+1]

# Parse MEME motif file
def parse_meme(filename):
    motifs = []
    with open(filename) as f:
        content = f.read()
    
    motif_blocks = re.split(r"(?m)^MOTIF ", content)
    header = motif_blocks[0]
    for block in motif_blocks[1:]:
        lines = block.strip().splitlines()
        motif_id = lines[0].split()[0]
        matrix_lines = []
        in_matrix = False
        for line in lines:
            if line.startswith("letter-probability matrix"):
                in_matrix = True
                continue
            if in_matrix:
                if not line.strip():
                    break
                matrix_lines.append([float(x) for x in line.strip().split()])
        motifs.append((motif_id, matrix_lines))
    return header, motifs

# Write MEME file back
def write_meme(header, motifs, out_file):
    with open(out_file, "w") as f:
        f.write(header.strip() + "\n")
        for motif_id, pwm in motifs:
            if not pwm:
                continue
            f.write(f"MOTIF {motif_id}\n")
            f.write("letter-probability matrix: alength= 4 w= {}\n".format(len(pwm)))
            for row in pwm:
                f.write(" ".join(f"{x:.6f}" for x in row) + "\n")
            f.write("\n")

# Save sequence logo as PNG
def save_logo(motif_id, pwm, threshold, out_prefix="logo"):
    df = pd.DataFrame(pwm, columns=['A','C','G','T'])
    ic_df = logomaker.transform_matrix(df, from_type='probability', to_type='information')
    
    logo = logomaker.Logo(ic_df, shade_below=0.5, fade_below=0.5, show_spines = False)

    # Remove y-axis
    ax = plt.gca()
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.spines['left'].set_visible(False)

    plt.title(f"")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_{motif_id}.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Trim motifs, filter by information content, and generate logos.")
    parser.add_argument("input", help="Input MEME motif file")
    parser.add_argument("output", help="Output MEME file with trimmed motifs")
    parser.add_argument("--threshold", type=float, default=0.2, help="Information threshold for trimming")
    parser.add_argument("--min_adj", type=int, default=5, help="Minimum consecutive informative positions")
    parser.add_argument("--logo_prefix", default="motif_logo", help="Prefix for motif logo PNG files")
    args = parser.parse_args()

    header, motifs = parse_meme(args.input)

    trimmed = []
    for motif_id, pfm in motifs:
        # Normalize columns to PWM (probabilities)
        pwm = []
        for row in pfm:
            col_sum = sum(row)
            pwm.append([x / col_sum for x in row])

        trimmed_pwm = trim_pwm(pwm, threshold=args.threshold)

        # --- FILTER STEP: keep only motifs with >= min_adj consecutive good columns ---
        infos = [column_information(col) for col in trimmed_pwm]
        max_adj = 0
        current = 0
        for val in infos:
            if val >= args.threshold:
                current += 1
                max_adj = max(max_adj, current)
            else:
                current = 0

        if max_adj >= args.min_adj:
            trimmed.append((motif_id, trimmed_pwm))
            save_logo(motif_id, trimmed_pwm, args.threshold, args.logo_prefix)

    write_meme(header, trimmed, args.output)

