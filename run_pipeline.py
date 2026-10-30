from src.train import main as train
from src.evaluate import main as evaluate
from src.adversarial_training import main as robust
if __name__=='__main__':
    train(); evaluate(); robust(); evaluate()
