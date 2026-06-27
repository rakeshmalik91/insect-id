import lief
import os
import shutil
import sys
import argparse

def patch_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.so'):
                path = os.path.join(root, file)
                try:
                    binary = lief.parse(path)
                    if not binary:
                        continue
                    patched = False
                    for segment in binary.segments:
                        if segment.type == lief.ELF.Segment.TYPE.LOAD:
                            if segment.alignment < 16384:
                                segment.alignment = 16384
                                patched = True
                    if patched:
                        binary.write(path)
                        print(f"Patched 16KB alignment for {path}")
                except Exception as e:
                    print(f"Failed to patch {path}: {e}")

def patch_aar(input_aar_path, output_aar_path):
    temp_dir = os.path.join(os.path.dirname(output_aar_path), "temp_aar_patch")
    
    print(f"Extracting {input_aar_path} to {temp_dir}...")
    os.makedirs(temp_dir, exist_ok=True)
    shutil.unpack_archive(input_aar_path, temp_dir, 'zip')

    print("Patching .so files inside...")
    patch_directory(temp_dir)

    print(f"Repackaging into {output_aar_path}...")
    
    # shutil.make_archive adds the extension automatically, so we remove .aar
    base_output = output_aar_path
    if base_output.endswith('.aar'):
        base_output = base_output[:-4]
        
    shutil.make_archive(base_output, 'zip', temp_dir)
    
    # Rename .zip to .aar
    if os.path.exists(base_output + ".zip"):
        if os.path.exists(output_aar_path):
            os.remove(output_aar_path)
        os.rename(base_output + ".zip", output_aar_path)
        
    print(f"Cleaning up {temp_dir}...")
    shutil.rmtree(temp_dir)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch native libraries in an AAR to be 16KB aligned for Android 15+.")
    parser.add_argument("input_aar", help="Path to the original .aar file")
    parser.add_argument("output_aar", help="Path to save the patched .aar file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_aar):
        print(f"Error: {args.input_aar} does not exist.")
        sys.exit(1)
        
    patch_aar(args.input_aar, args.output_aar)
