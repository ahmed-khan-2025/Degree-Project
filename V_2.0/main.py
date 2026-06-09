import os

print("Training PPO...")
os.system("python training/train_ppo.py")

print("\nEvaluating...")
os.system("python evaluation/evaluate.py")

print("\nDONE")