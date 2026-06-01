"""CLI helpers for FADE-Net training — shared between argparse and interactive menu."""

import argparse


def default_args(**overrides):
    """Single source of truth for training argument defaults.
    Used by both the interactive menu and batch subprocess fallback."""
    defaults = dict(
        seed=42, epochs=None, batch_size=None, split=None,
        freeze=None, resume=False, fresh=False, overwrite_artifacts=False,
        backbone_source=None, backbone_name=None, no_pretrained=False,
        afad_dir=None, allow_legacy_split_upgrade=False,
        max_train_batches=None, max_val_batches=None, max_test_batches=None,
        experiment_tag=None, split_file_tag="formal_v1",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def interactive_menu(run_training_fn, run_batch_fn):
    """Interactive training menu. Delegates to run_training_fn(seed_args) or run_batch_fn(seed)."""
    print("=" * 60)
    print("🎮 FADE-Net Interactive Training Launcher")
    print("=" * 60)
    print("1. [Default]  Run Standard Benchmark (Seed 42, 72-8-20, formal_v1)")
    print("2. [Seed]     Run 2026 Academic Seed (Seed 2026, 72-8-20, formal_v1)")
    print("3. [Batch]    Run Academic Seeds (42, 3407, 2026, formal_v1)")
    print("4. [Custom]   Configure Manually")
    print("q. [Quit]     Exit")
    print("-" * 60)

    try:
        choice = input("👉 Select mode [1-4/q]: ").strip().lower()

        if choice == '1' or choice == '':
            print("\n🚀 Selected: Standard Benchmark (Seed 42)")
            run_training_fn(default_args())

        elif choice == '2':
            print("\n🚀 Selected: Academic Seed 2026")
            run_training_fn(default_args(seed=2026))

        elif choice == '3':
            print("\n🚀 Selected: Run All Academic Seeds")
            import numpy as np
            seeds = [42, 3407, 2026]
            results = {}
            for s in seeds:
                mae = run_batch_fn(s)
                if mae is not None:
                    results[s] = mae
            print("\n" + "=" * 60)
            print("📊 Final Batch Report")
            print("=" * 60)
            if results:
                maes = list(results.values())
                mean_mae = np.mean(maes)
                std_mae = np.std(maes)
                print(f"{'Seed':<10} | {'Test MAE':<10}")
                print("-" * 25)
                for s, m in results.items():
                    print(f"{s:<10} | {m:.4f}")
                print("-" * 25)
                print(f"\n🏆 Average Test MAE: {mean_mae:.4f} ± {std_mae:.4f}")
            else:
                print("No successful runs.")

        elif choice == '4':
            print("\n🔧 Custom Configuration Mode:")
            s = input("   - Seed [42]: ").strip() or '42'
            sp_choice = input("   - Split (1: 72-8-20, 2: 80-10-10, 3: 90-5-5) [1]: ").strip()
            if sp_choice == '2':
                split = '80-10-10'
            elif sp_choice == '3':
                split = '90-5-5'
            else:
                split = '72-8-20'
            ep = input("   - Epochs [Default]: ").strip()
            fz = input("   - Freeze Epochs [Default]: ").strip()
            run_training_fn(default_args(
                seed=int(s), split=split,
                epochs=int(ep) if ep else None,
                freeze=int(fz) if fz else None,
            ))

        elif choice == 'q':
            pass

    except KeyboardInterrupt:
        print("\n👋 Exiting.")
        import sys
        sys.exit(0)
