import os
import zipfile
import numpy as np
from tqdm import tqdm
from main import CollisionBoundaryPatternFormation

# Configuration from multiRun.py
SAVE_PATH = r"F:\MS_ExperimentData\general"

# Parameters (must match multiRun.py)
phaseLags = [0 * np.pi]
omegaMins = [0]
randomSeeds = [10]
strengthKs = np.linspace(1, 20, 7)
distanceD0s = np.linspace(0.1, 3, 7)
deltaOmegas = [0]

def compress_by_parameter_group():
    """
    Compress files grouped by phaseLags.
    """
    total_groups = len(phaseLags)
    print(f"Starting compression grouped by phaseLags (Total groups: {total_groups})")

    if not os.path.exists(SAVE_PATH):
        print(f"Error: SAVE_PATH does not exist: {SAVE_PATH}")
        return

    # Loop 1: phaseLag (The Grouping Parameter)
    for i, phaseLag in enumerate(tqdm(phaseLags, desc="Processing Groups")):
        
        # Collect all files for this phaseLag
        group_files = []
        
        # Inner loops for the current phaseLag
        for strengthK in strengthKs:
            for distanceD0 in distanceD0s:
                for omegaMin in omegaMins:
                    for deltaOmega in deltaOmegas:
                        for randomSeed in randomSeeds:
                            # Instantiate model to get the exact filename
                            model = CollisionBoundaryPatternFormation(
                                strengthK=strengthK, distanceD0=distanceD0, phaseLagA0=phaseLag,
                                freqDist="uniform", initPhaseTheta=None,
                                omegaMin=omegaMin, deltaOmega=deltaOmega, 
                                agentsNum=2000, dt=0.005,
                                tqdm=True, savePath=SAVE_PATH, shotsnaps=10, 
                                randomSeed=randomSeed, overWrite=False
                            )
                            filename = f"{str(model)}.h5"
                            file_path = os.path.join(SAVE_PATH, filename)
                            group_files.append(file_path)

        # Process the group
        if not group_files:
            continue
            
        # Check if any files in this group actually exist
        existing_files = [f for f in group_files if os.path.exists(f)]
        
        if not existing_files:
            # print(f"  No existing files found for phaseLag={phaseLag:.3f}, skipping.")
            continue

        zip_name = f"data_phaseLag_{i}_{phaseLag:.3f}.zip"
        zip_path = os.path.join(SAVE_PATH, zip_name)
        
        print(f"\nCompressing group {i+1}/{total_groups}: phaseLag={phaseLag:.3f} -> {zip_name}")
        print(f"  Found {len(existing_files)} files to compress.")

        try:
            # 1. Compress
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in existing_files:
                    zipf.write(file_path, arcname=os.path.basename(file_path))
            
            # 2. Verify
            # print("  Verifying zip archive...")
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                bad_file = zipf.testzip()
                if bad_file:
                    raise Exception(f"Corrupt file in zip: {bad_file}")
            
            # 3. Delete original files
            # print("  Deleting original files...")
            for file_path in existing_files:
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f"    Warning: Could not delete {os.path.basename(file_path)}: {e}")
            
            print(f"  Successfully processed group {i+1}.")

        except Exception as e:
            print(f"  Error processing group {i+1}: {e}")
            if os.path.exists(zip_path):
                print(f"  Removing potentially corrupt zip: {zip_path}")
                os.remove(zip_path)
            # Do not delete source files if compression failed

if __name__ == "__main__":
    compress_by_parameter_group()
