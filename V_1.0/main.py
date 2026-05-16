import os

print("Training PPO...")
os.system("python train/train.py")

print("\nEvaluating...")
os.system("python eval/evaluate.py")

print("\nDONE")